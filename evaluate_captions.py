import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import nltk
from nltk.translate.bleu_score import sentence_bleu
from pycocoevalcap.spice.spice import Spice
import json
from rouge import Rouge 
import evaluate
from transformers.image_utils import load_image
import copy
from argparse import ArgumentParser

torch.set_float32_matmul_precision('high')
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True


facilities='''
Accessible entrances/door
Accessible pathway/wheelchair ramps
Directional guide paths
Warning guide paths
Dropped kerbs
Accessible lifts/elevators
Accessible signage
Braille and tactile floor plans/maps
Accessible carparks
Audible/visual signaling devices
People with disabilities
'''

PROMPT_CAPTION_TEMPLATE=f'''
Inside this image there exists facilities designed for the disabilities. Now please search over the image and LIST each of the facilities you find within 6 words. Seperate each of the description with comma. 

Possible facilities are:

{facilities}

What related facilities do you find in the image?
'''

PALIGEMMA_CAPTION_PROMPT=f'''
    caption What facilities designed for the disabilities do you find in the image? 
'''


class CaptionEvaluator:
    def __init__(self, model_id="google/paligemma-3b-pt-448", 

    ):
        self.device = "cuda"
        self.is_chat_model = False
        # self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()
        # self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        if "Qwen2.5" in model_id:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

            self.is_chat_model = True
        
        if "paligemma" in model_id:
            if "chat" not in model_id.lower():
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.bfloat16,).to(self.device).eval()
                self.processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)
                self.is_chat_model = False
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
                self.is_chat_model=True



        self.rouge = Rouge()
        self.google_bleu=evaluate.load("google_bleu")
        self.meteor=evaluate.load("meteor")
        self.bertscore=evaluate.load("bertscore")
        self.exact_match=evaluate.load("exact_match")
    
    def build_caption_query(self, image_dict):
        pbar = tqdm(image_dict.items(), desc="Building queries with chat template")
        queries=[]
        for key, value in pbar:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": value["image_path"],
                        },
                        {"type": "text", "text": PROMPT_CAPTION_TEMPLATE},
                    ],
                }
            ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            queries.append((text, image_inputs, video_inputs, value["caption"]))
        return queries

    def calc_metrics(self, caption, gt_caption):
        # Store original values in result
        result = {
            "ground_truth": gt_caption,
            "prediction": caption
        }
        
        # Make copies for metric calculation to avoid modifying originals
        caption_copy = copy.deepcopy(caption)
        gt_caption_copy = copy.deepcopy(gt_caption)
        
        # Calculate bert_score and rouge_score using joined strings
        bert_score = self.bertscore.compute(
            predictions=[" ".join(caption)], 
            references=[" ".join(gt_caption)], 
            lang="en", 
            use_fast_tokenizer=True, 
            device="cuda:0"
        )
        rouge_score = self.rouge.get_scores(
            hyps=" ".join(caption), 
            refs=" ".join(gt_caption)
        )
        
        # Ensure copies have same length for other metrics
        max_len = max(len(caption_copy), len(gt_caption_copy))
        if len(caption_copy) < max_len:
            caption_copy.extend([caption_copy[-1]] * (max_len - len(caption_copy)))
        elif len(gt_caption_copy) < max_len:
            gt_caption_copy.extend([gt_caption_copy[-1]] * (max_len - len(gt_caption_copy)))
            
        # Calculate other metrics using padded copies
        google_bleu_score = self.google_bleu.compute(
            predictions=caption_copy, 
            references=gt_caption_copy
        )
        meteor_score = self.meteor.compute(
            predictions=caption_copy, 
            references=gt_caption_copy
        )
        exact_match_score = self.exact_match.compute(
            predictions=caption_copy, 
            references=gt_caption_copy
        )
        
        result.update({
            "exact_match": exact_match_score["exact_match"].item(),
            "google_bleu": google_bleu_score["google_bleu"],
            "meteor": meteor_score["meteor"].item(),
            "rouge_score": rouge_score,
            "bert_score": bert_score
        })
        return result
    
    def evaluate_chat_model(self, image_dict):
        results = dict()
        queries = self.build_caption_query(image_dict)
        assert len(queries)==len(image_dict), f"Queries: {len(queries)}, Images: {len(image_dict)}"
        pbar = tqdm(range(len(queries)), desc="Evaluating")
        # spice = Spice()
        
        for i in pbar:
            text, image_inputs, video_inputs, gt_caption = queries[i]
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            caption = output_text[0]
            results[i] = self.calc_metrics(caption.split(", "), gt_caption)
            # spice_score = spice.compute_score({i:["\n".join(gt_caption)]}, {i:[caption]})
            # bleu_score=sentence_bleu("\n".join(gt_caption).split(), caption.split())
            # rouge_score = rouge.compute(predictions=[caption], references=["\n".join(gt_caption)])
            
            
        return results

    def evaluate_model(self, image_dict):
        prompt = PALIGEMMA_CAPTION_PROMPT
        pbar = tqdm(range(len(image_dict)))
        results=dict()
        for i in pbar:
            
            image = Image.open(image_dict[str(i)]["image_path"]).convert("RGB")
            model_inputs = self.processor(text="<image>\n"+prompt,padding=True, images=image, return_tensors="pt").to(torch.bfloat16).to(self.device)
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)

            results[i] = self.calc_metrics(decoded, image_dict[str(i)]["caption"])

        return results
    
    def evaluate_paligemma_chat(self, image_dict):

        return

def main(args):
    # Example usage
    
    # model_id="BUAADreamer/PaliGemma-3B-Chat-v0.2"
    # model_id="google/paligemma-3b-ft-vqav2-448"
    model_id=args.model_id
    with open(args.image_dict_path, "r") as f:
        image_dict = json.load(f)
    
    # evaluator = CaptionEvaluator(
    #     model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    # )
    # results = evaluator.evaluate_chat_model(image_dict)
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(results, f, indent=4)
    
    evaluator=CaptionEvaluator(
        model_id
    )
    # results = evaluator.evaluate_model(image_dict)
    if args.use_chat_model:
        results = evaluator.evaluate_chat_model(image_dict)
    with open(f"{model_id.split('/')[-1]}_caption_results_{args.run_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    
    # Print results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_dict_path", default="filtered_images_test_labeled.json")
    parser.add_argument("--run_name", default="Name to save results", required=True)
    parser.add_argument("--use_chat_model", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
