import datasets
from datasets import Dataset, load_dataset
import torch
from v2stoks import TTSemanticToken
import pdb



def process_text(dataset: Dataset, tts_semantic ):
    for i in range(100):
        sample = dataset[i]
        text = sample['conversations']
        inputs = []
        outputs = []
        for entry in text:
            if entry['from'] == 'human':
                image, context = entry['value'].split('<image>')
                context = context.strip()
                semantics = tts_semantic.convert_text_to_semantic(context)
                if i % 5 == 0:
                    print(context)
                    print(semantics[0][0])
                    print("Semantic Tokens Shape:", semantics.shape)
                    print("Semantic Tokens datatype:", semantics.dtype)
                    print("Device:", semantics.device)
    

if __name__ == '__main__':
    dataset = load_dataset("lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1", split = "caption" )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_semantic = TTSemanticToken(device)
    process_text(dataset, tts_semantic)
       
    
    




