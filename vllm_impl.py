import torch
from transformers import AutoProcessor

from vllm import LLM, SamplingParams

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


class CaptionEvaluator:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", batch_size=4
    ):
        if "Qwen2.5" in model_id:
            self.model=LLM(model=model_id,
                           trust_remote_code=True,
                           max_model_len=32768,
                           limit_mm_per_prompt={"image": 1},
                        )
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            self.batch_size = batch_size
        else:
            # Fallback or error for unsupported models
            raise ValueError(f"Unsupported model_id for this script: {model_id}. Only Qwen2.5-VL models are supported.")
        
       



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
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            queries.append((text, image_inputs, video_inputs, value["caption"], video_kwargs))
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
        # build_caption_query prepares (text, image_inputs, video_inputs, gt_caption, video_kwargs) for each item
        # We'll adapt this logic to build a list of llm_inputs directly.
        queries_data = self.build_caption_query(image_dict) # This still gives us all components
        assert len(queries_data) == len(image_dict), f"Queries data count: {len(queries_data)}, Images: {len(image_dict)}"

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=128, 
            stop_token_ids=self.processor.tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_end|>"]) if hasattr(self.processor, 'tokenizer') else [],
        )

        all_llm_inputs = []
        all_ground_truths = []

        pbar_prepare = tqdm(queries_data, desc="Preparing all inputs")
        for query_data_item in pbar_prepare:
            text, image_inputs, video_inputs, gt_caption, video_kwargs = query_data_item
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            # Assuming video_inputs will be None for this image captioning task based on build_caption_query

            llm_input_item = {
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }
            all_llm_inputs.append(llm_input_item)
            all_ground_truths.append(gt_caption)

        if not all_llm_inputs:
            print("No inputs to process.")
            return results

        # Process in batches to avoid OOM
        all_outputs = []
        total_batches = (len(all_llm_inputs) + self.batch_size - 1) // self.batch_size
        
        pbar_generate = tqdm(range(total_batches), desc=f"Generating captions (batch_size={self.batch_size})")
        for batch_idx in pbar_generate:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_llm_inputs))
            
            batch_inputs = all_llm_inputs[start_idx:end_idx]
            batch_outputs = self.model.generate(batch_inputs, sampling_params)
            all_outputs.extend(batch_outputs)
        
        # Extract all captions before deleting the model
        all_captions = []
        for i, generated_output in enumerate(all_outputs):
            if generated_output.outputs:
                caption = generated_output.outputs[0].text.strip()
            else:
                caption = ""
                print(f"Warning: No output generated for input index {i}")
            all_captions.append(caption)
        
        # Delete the model to free up GPU memory before metrics calculation
        print("Deleting VLM model to free up GPU memory...")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pbar_metrics = tqdm(range(len(all_captions)), desc="Calculating metrics")
        for i in pbar_metrics:
            caption = all_captions[i]
            gt_caption = all_ground_truths[i]
            
            # Assuming results keys are 0-indexed integers corresponding to the order in image_dict
            results[i] = self.calc_metrics(caption.split(", "), gt_caption)
            
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
        model_id,
        batch_size=args.batch_size
    )
    results = evaluator.evaluate_chat_model(image_dict)
    with open(f"{model_id.split('/')[-1]}_caption_results_{args.run_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    
    # Print results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_dict_path", default="filtered_images_test_labeled.json")
    parser.add_argument("--run_name", default="Name to save results", required=True)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing inputs")
    args = parser.parse_args()
    main(args)
