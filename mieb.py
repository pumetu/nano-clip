import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Any

import open_clip
import torch
from tqdm import tqdm

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, PromptType
from mteb.cache import ResultCache

class OpenCLIPModel(AbsEncoder):
    def __init__(self, model_name: str, pretrained: str = None, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model, _, self.img_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_text_embeddings(self, texts, show_progress_bar: bool = True, **kwargs):
        all_text_embeddings = []
        with torch.no_grad():  # , torch.cuda.amp.autocast():
            for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
                inputs = self.tokenizer(batch["text"])
                text_outputs = self.model.encode_text(inputs.to(self.device))
                all_text_embeddings.append(text_outputs.cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(self, images, show_progress_bar: bool = True, **kwargs):
        all_image_embeddings = []
        with torch.no_grad():  # , torch.cuda.amp.autocast():
            for batch in tqdm(images, disable=not show_progress_bar, desc="Image Encoding"):
                inputs = torch.vstack([self.img_preprocess(b).unsqueeze(0) for b in batch["image"]])
                image_outputs = self.model.encode_image(inputs.to(self.device))
                all_image_embeddings.append(image_outputs.cpu())
        return torch.cat(all_image_embeddings, dim=0)

    def encode(
        self,
        inputs,
        *,
        task_metadata: TaskMetadata = None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError("The number of texts and images must have the same length")
            return text_embeddings + image_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image features found in input")


def main(args):
    model = OpenCLIPModel(model_name=args.model, pretrained=args.pretrained)
    model_name = Path(args.pretrained).parent.parent.name
    model.mteb_model_meta = ModelMeta(
        loader=None,
        name="nano-clip/" + model_name,
        revision=None,
        release_date="2026-01-01",  # dummy date
        languages=["eng-Latn"],
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=77,
        embed_dim=512,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch"],
        similarity_fn_name=None,
        modalities=["text", "image"],
        training_datasets=None,
        license=None,
        open_weights=None,
        use_instructions=False,
    )
    if len(args.benchmark) == 1 and "mieb" in args.benchmark[0].lower():
        tasks = mteb.get_benchmarks(args.benchmark)[0].tasks
    else:
        tasks = mteb.get_tasks(task_types=args.benchmark)

    root = os.path.join(args.root_folder, model_name)
    mteb.evaluate(
        model,
        tasks=tasks,
        num_proc=args.num_proc,
        prediction_folder=os.path.join(root, "prediction"),
        cache=ResultCache(root),
    )

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate MIEB")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--benchmark", type=str, nargs="+", default=["MIEB(eng)"])
    parser.add_argument("--root-folder", type=str, default="mieb")
    parser.add_argument("--num-proc", type=int, default=16)
    main(parser.parse_args())
