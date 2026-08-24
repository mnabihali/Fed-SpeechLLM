import torch
import argparse
import numpy as np
import random
import warnings
from pytorch_lightning import Trainer
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
import torch.utils.data as data_utils
import os

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # -----------------------------
    # Deterministic setup
    # -----------------------------
    set_seed(42)

    # -----------------------------
    # Argument parser
    # -----------------------------
    parser = argparse.ArgumentParser(description="Test SpeechLLM with specific checkpoint round.")
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="Checkpoint round number to load (e.g., 9 for Checkpoint-round-9.ckpt)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="path for the csv file"
    )
    args = parser.parse_args()

    # -----------------------------
    # Model configuration
    # -----------------------------
    model_config = {
        'audio_enc_dim': 1024,
        'llm_dim': 2048,
        'audio_encoder_name':"openai/whisper-medium", # "microsoft/wavlm-large",
        'connector_name': 'linear',
        'llm_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        'finetune_encoder': False,
        'connector_k': 2,
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 16,
        'max_lr': 3e-4,
        'total_training_step': 1000000,
        'warmup_steps': 100,
        'train_batch_per_epoch': 200,
        'grad_accumulate_steps': 8
    }

    # -----------------------------
    # Initialize model
    # -----------------------------
    model = SpeechLLMLightning(**model_config)

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    checkpoint_path=f"/stek/mohamed/FL-SLAM/6lang_allLang_server_finetuning/round-{args.round}.ckpt" 
    print(f"\nLoading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded and set to evaluation mode ?")

    # -----------------------------
    # Log trainable parameters
    # -----------------------------
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for _, p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())

    print("\n=== Model Parameter Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable:,}")
    print(f"Frozen parameters: {total_params - total_trainable:,}\n")

    print("Trainable layers:")
    for name, param in trainable_params:
        print(f"  {name}: {param.numel():,}")

    # Create directory for logs
    os.makedirs("parameter_logs", exist_ok=True)
    log_path = os.path.join("parameter_logs", f"trainable_parameters_round_{args.round}.txt")

    # Write summary to text file
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {total_trainable:,}\n")
        f.write(f"Frozen parameters: {total_params - total_trainable:,}\n\n")
        f.write("Trainable layers:\n")
        for name, param in trainable_params:
            f.write(f"{name}: {param.numel():,}\n")

    print(f"\n? Trainable parameter summary saved to: {log_path}\n")

    # -----------------------------
    # Dataset and DataLoader setup
    # -----------------------------
    tokenizer = model.llm_tokenizer
    test_dataset = InstructionalAudioDataset(
        csv_file=  args.csv, #"/stek/mohamed/FL-SLAM/csv_mlsitalian/test_itmls.csv", # "/stek/mohamed/FL-SLAM/test.csv", #
        mode='test'
    )
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)

    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=my_collator,
        num_workers=0,  # deterministic
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # -----------------------------
    # Run deterministic test
    # -----------------------------
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        deterministic=True
    )

    print(f"\n?? Starting test for checkpoint round {args.round}...\n")
    trainer.test(model=model, dataloaders=test_loader)
    print("\n Test complete! \n")



'''
import torch
import argparse
import numpy as np
import random
import warnings
from pytorch_lightning import Trainer
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
import torch.utils.data as data_utils

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Set seed before everything else for deterministic results
    set_seed(42)
    
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Test SpeechLLM with specific checkpoint round.")
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="Checkpoint round number to load (e.g., 9 for Checkpoint-round-9.ckpt)"
    )
    args = parser.parse_args()
    
    # -----------------------------
    # Model configuration
    # -----------------------------
    model_config = {
        'audio_enc_dim': 1024,
        'llm_dim': 2048,
        'audio_encoder_name': "microsoft/wavlm-large",
        'connector_name': 'linear',
        'llm_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        'finetune_encoder': False,
        'connector_k': 2,
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 16,
        'max_lr': 3e-4,
        'total_training_step': 1000000,
        'warmup_steps': 100,
        'train_batch_per_epoch': 200,
        'grad_accumulate_steps': 8
    }
    
    # -----------------------------
    # Initialize model
    # -----------------------------
    model = SpeechLLMLightning(**model_config)
    
    # -----------------------------
    # Load checkpoint dynamically
    # -----------------------------
    checkpoint_path = f"/stek/mohamed/FL-SLAM/FL_SLAM_checkpoints/Checkpoint-round-{args.round}.ckpt"
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Explicitly set model to evaluation mode
    model.eval()
    print("Model set to evaluation mode")
    
    # -----------------------------
    # Dataset and DataLoader setup
    # -----------------------------
    tokenizer = model.llm_tokenizer
    test_dataset = InstructionalAudioDataset(
        csv_file="/stek/mohamed/SpeechLLM/test.csv",
        mode='test'
    )
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    
    # Use single worker for deterministic results
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=my_collator,
        num_workers=0,  # Changed from 3 to 0 for determinism
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # -----------------------------
    # Run test with deterministic settings
    # -----------------------------
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        deterministic=True  # Enable deterministic mode
    )
    
    print(f"Starting test for checkpoint round {args.round}...")
    trainer.test(model=model, dataloaders=test_loader)

'''
'''
import torch
import argparse
from pytorch_lightning import Trainer
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
import torch.utils.data as data_utils

if __name__ == "__main__":
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Test SpeechLLM with specific checkpoint round.")
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="Checkpoint round number to load (e.g., 9 for Checkpoint-round-9.ckpt)"
    )
    args = parser.parse_args()

    # -----------------------------
    # Model configuration
    # -----------------------------
    model_config = {
        'audio_enc_dim': 1024,
        'llm_dim': 2048,
        'audio_encoder_name': "microsoft/wavlm-large",
        'connector_name': 'linear',
        'llm_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        'finetune_encoder': False,
        'connector_k': 2,
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 16,
        'max_lr': 3e-4,
        'total_training_step': 1000000,
        'warmup_steps': 100,
        'train_batch_per_epoch': 200,
        'grad_accumulate_steps': 8
    }

    # -----------------------------
    # Initialize model
    # -----------------------------
    model = SpeechLLMLightning(**model_config)

    # -----------------------------
    # Load checkpoint dynamically
    # -----------------------------
    checkpoint_path = f"/stek/mohamed/FL-SLAM/FL_SLAM_checkpoints/Checkpoint-round-{args.round}.ckpt"
    print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # -----------------------------
    # Dataset and DataLoader setup
    # -----------------------------
    tokenizer = model.llm_tokenizer
    test_dataset = InstructionalAudioDataset(
        csv_file="/stek/mohamed/SpeechLLM/test.csv",
        mode='test'
    )
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=my_collator,
        num_workers=3
    )

    # -----------------------------
    # Run test
    # -----------------------------
    trainer = Trainer(accelerator='gpu', devices=1)
    trainer.test(model=model, dataloaders=test_loader)

'''


'''
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset
import torch.utils.data as data_utils
from dataset import InstructionalAudioDataset, MyCollator

if __name__ == "__main__":
    
    model_config = {
                'audio_enc_dim': 1024, 
                'llm_dim': 2048, 
                'audio_encoder_name': "microsoft/wavlm-large",
                'connector_name': 'linear',
                'llm_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                'finetune_encoder': False,
                'connector_k': 2,
                'use_lora': True,
                'lora_r': 8,
                'lora_alpha': 16,
                'max_lr': 3e-4,
                'total_training_step': 1000000,
                'warmup_steps': 100,
                'train_batch_per_epoch': 200,
                'grad_accumulate_steps': 8
        }  
    
    # Create model instance first
    model = SpeechLLMLightning(**model_config)
    
    # Load the state_dict from checkpoint
    checkpoint_path = "/stek/mohamed/FL-SLAM/FL_SLAM_checkpoints/Checkpoint-round-9.ckpt"
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    
    # Get tokenizer
    tokenizer = model.llm_tokenizer
    
    # Load test dataset
    test_dataset = InstructionalAudioDataset(
        csv_file="/stek/mohamed/SpeechLLM/test.csv",
        mode='test'
    )
    
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collator, num_workers=3)
    
    # Create trainer and test
    trainer = Trainer(
        accelerator='gpu', devices=1
    )
    trainer.test(model=model, dataloaders=test_loader)
'''
