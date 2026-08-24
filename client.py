from collections import OrderedDict
import psutil
import numpy as np
import torch
import flwr as fl
import wandb
import random
import gc
from pytorch_lightning import Trainer
from trainer import SpeechLLMLightning
from dataset import MyCollator, build_dataloaders_from_csvs
from pytorch_lightning.strategies import DDPStrategy
from typing import Dict, List, Optional, Tuple, Callable, Union
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (Code, EvaluateRes, EvaluateIns, FitRes, FitIns, GetParametersRes, GetParametersIns, Status,
                         Scalar, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays)

import warnings
warnings.filterwarnings('ignore')

# ---------> Seed Fix <---------
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
fix_seed(12344) #

# ---------> Model & Model Config <---------

model_config = {
    'audio_enc_dim': 1024,
    'llm_dim': 2048,
    'audio_encoder_name': "openai/whisper-medium", #"microsoft/wavlm-large",
    'connector_name': 'linear',
    'llm_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    'finetune_encoder': False,
    'connector_k': 2,
    'use_lora': True,
    'lora_r': 8,
    'lora_alpha': 16,
    'max_lr': 1e-4,
    'total_training_step': 10000000,
    'warmup_steps': 100,
    'train_batch_per_epoch': 200,
    'grad_accumulate_steps': 4
}

model = SpeechLLMLightning(**model_config)
tokenizer = model.llm_tokenizer

# ---------> Dataset Loading <---------

my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
csv_train_dir =  "./csvs_multilingual", #"flieurs_mt_clients" #"./fl_MLS_train_speaker" #"./fl_multilingual" #"./fl_MLS_train_speaker" #"./fl_LS_train100_speaker"
csv_dev_dir = "./fl_MLS_dev_speaker" #"./fl_LS_dev_speaker"
train_loaders = build_dataloaders_from_csvs(csv_dir=csv_train_dir, my_collator=my_collator,
                                            batch_size=4, num_workers=3, shuffle=True)
dev_loaders = build_dataloaders_from_csvs(csv_dir=csv_dev_dir, my_collator=my_collator,
                                          batch_size=1, num_workers=3, shuffle=False)


# ---------> Set & Get parameters <---------

def get_trainable_parameters(model) -> Tuple[List[np.ndarray], List[str]]:
    """Extract only trainable parameters and their names."""
    trainable_params = []
    trainable_names = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param.detach().cpu().numpy())
            trainable_names.append(name)

    return trainable_params, trainable_names


def get_parameters(model) -> List[np.ndarray]:
    """Extract only trainable model parameters as a list of NumPy arrays."""
    trainable_params, _ = get_trainable_parameters(model)
    return trainable_params


def set_parameters(model, parameters: List[np.ndarray]):
    """Load only trainable parameters into the model."""
    # Get names of trainable parameters
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    if len(parameters) != len(trainable_names):
        raise ValueError(f"Expected {len(trainable_names)} parameters, got {len(parameters)}")

    # Create state dict with only trainable parameters
    params_dict = zip(trainable_names, parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )

    # Load with strict=False since we're only updating trainable parameters
    model.load_state_dict(state_dict, strict=False)


# ---------> Aggregation Strategy <---------

class CustomStrategy(fl.server.strategy.FedAvg):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"

       
#    def initialize_parameters(
#        self, client_manager: ClientManager
#    ) -> Optional[Parameters]:
#        """Initialize global model parameters."""
#
#        #state_dict = torch.load("./pretrained_models/librispeech1000.pth")
#        state_dict = torch.load("/stek/mohamed/FL-SLAM/checkpoints_multilingual_many_languages/round-134.ckpt") #modification
#        #next lines if we want to use checkpoints... Must be parameterized.
#        #list_of_files = [fname for fname in glob.glob("./trained_models/round-0*")]
#        #latest_round_file = max(list_of_files, key=os.path.getctime)
#        #print("Loading pre-trained model from: ", latest_round_file)
#        #state_dict = torch.load(latest_round_file)
#        model.load_state_dict(state_dict)
#        torch.cuda.empty_cache()
#        #gc.collect()
#        ndarrays = get_parameters(model)
#        print("---- Done loading Checkpoints ----")
#        print("---- Resume form Checkpoints ----")
#        return fl.common.ndarrays_to_parameters(ndarrays)
#   

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
    
        # Sample clients
        #sample_size, min_num_clients = self.num_fit_clients(
            #client_manager.num_available()
        #)
        #clients = client_manager.sample(  # ? Removed random sampling
            #num_clients=sample_size, min_num_clients=min_num_clients
        #)
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size = random.randint(min_num_clients, sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
    
        # ? Learning rate decay: reduce by 10% every 10 rounds
        initial_lr = 0.001
        decay_factor = 0.9
        decay_every = 10
        current_lr = initial_lr * (decay_factor ** (server_round // decay_every))
        
        print(f"[Round {server_round}] Learning rate: {current_lr:.6f}")
    
        # ? All clients use the same decaying learning rate
        config = {"lr": current_lr, "local_epochs": 10}
        fit_configurations = [
            (client, FitIns(parameters, config)) for client in clients
        ]
        
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate updated model weights from clients."""
    
        if not results:
            print(f"[Round {server_round}] ? No results received from clients.")
            return None, {}
    
        if self.accept_failures and failures:
            print(f"[Round {server_round}] ?? Some clients failed, skipping aggregation.")
            return None, {}
    
        # ---- Convert results to numpy arrays ----
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
    
        # ---- Weighted average aggregation ----
        # Each client's contribution is proportional to its dataset size
        total_examples = sum(num_examples for _, num_examples in weights_results)
        if total_examples == 0:
            print(f"[Round {server_round}] ?? No training examples reported.")
            return None, {}
    
        # Compute weighted average of model updates
        aggregated_weights = aggregate(weights_results)

    
        # ---- Update global model (only trainable params) ----
        trainable_names = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
    
        if len(aggregated_weights) != len(trainable_names):
            print(
                f"[Round {server_round}] ?? Mismatch: "
                f"{len(aggregated_weights)} aggregated tensors vs {len(trainable_names)} trainable params."
            )
    
        # Build state dict for trainable parameters only
        params_dict = zip(trainable_names, aggregated_weights)
        state_dict = OrderedDict(
            {k: torch.tensor(np.array(v)) for k, v in params_dict}
        )
    
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(f"[Round {server_round}] ? Error loading state_dict: {e}")
    
        # ---- Save model checkpoint ----
        import os, gc
        os.makedirs("monolingual_maltese_fleurs", exist_ok=True)
        #mu = server_round+134
        ckpt_path = f"monolingual_maltese_fleurs/round-{server_round}.ckpt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Round {server_round}] ? Aggregated model saved at {ckpt_path}")
    
        # ---- Return new parameters ----
        new_parameters = get_parameters(model)
        del results, weights_results
        gc.collect()
    
        return ndarrays_to_parameters(new_parameters), {}

'''
class CustomStrategy(fl.server.strategy.FedAvg):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"

      
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        #state_dict = torch.load("./pretrained_models/librispeech1000.pth")
        state_dict = torch.load("/stek/mohamed/Wavlm_adapt_FL/FL_WAVLM_conv/wavlm-adapt-round-15.pth") #modification
        #next lines if we want to use checkpoints... Must be parameterized.
        #list_of_files = [fname for fname in glob.glob("./trained_models/round-0*")]
        #latest_round_file = max(list_of_files, key=os.path.getctime)
        #print("Loading pre-trained model from: ", latest_round_file)
        #state_dict = torch.load(latest_round_file)
        net.load_state_dict(state_dict)
        torch.cuda.empty_cache()
        #gc.collect()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)
    

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size = random.randint(min_num_clients, sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate updated model weights from clients."""
    
        if not results:
            print(f"[Round {server_round}] ? No results received from clients.")
            return None, {}
    
        if self.accept_failures and failures:
            print(f"[Round {server_round}] ?? Some clients failed, skipping aggregation.")
            return None, {}
    
        # ---- Convert results to numpy arrays ----
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
    
        # ---- Weighted average aggregation ----
        # Each client's contribution is proportional to its dataset size
        total_examples = sum(num_examples for _, num_examples in weights_results)
        if total_examples == 0:
            print(f"[Round {server_round}] ?? No training examples reported.")
            return None, {}
    
        # Compute weighted average of model updates
        aggregated_weights = aggregate(weights_results)

    
        # ---- Update global model (only trainable params) ----
        trainable_names = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
    
        if len(aggregated_weights) != len(trainable_names):
            print(
                f"[Round {server_round}] ?? Mismatch: "
                f"{len(aggregated_weights)} aggregated tensors vs {len(trainable_names)} trainable params."
            )
    
        # Build state dict for trainable parameters only
        params_dict = zip(trainable_names, aggregated_weights)
        state_dict = OrderedDict(
            {k: torch.tensor(np.array(v)) for k, v in params_dict}
        )
    
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(f"[Round {server_round}] ? Error loading state_dict: {e}")
    
        # ---- Save model checkpoint ----
        import os, gc
        os.makedirs("FL_SLAM_checkpoints", exist_ok=True)
        ckpt_path = f"FL_SLAM_checkpoints/Checkpoint-round-{server_round}.ckpt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Round {server_round}] ? Aggregated model saved at {ckpt_path}")
    
        # ---- Return new parameters ----
        new_parameters = get_parameters(model)
        del results, weights_results
        gc.collect()
    
        return ndarrays_to_parameters(new_parameters), {}
'''

# ---------> SLAM-ASR client <---------

class SpeechLLMClient(fl.client.NumPyClient):
    """ Federated learning Client using SpeechLLM & Flower"""

    def __init__(self, cid, model, train_loader, val_loader, model_config):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """ Return current model parameters """
        print(f"[Client {self.cid} get parameters]")
        return get_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """ Update model parameters """
        set_parameters(self.model, parameters)
    
    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """ Model Training Locally """
        print(f"[Client {self.cid}] Start Training...")  # ? Better logging
        # Set parameters received from the server
        self.set_parameters(parameters)
    
        # Get training config
        lr = config.get("lr", self.model_config['max_lr'])
        local_epochs = config.get("local_epochs", 10)
        
        # ? CRITICAL: Update model's learning rate (this makes the LR actually work!)
        self.model.max_lr = lr
        print(f"[Client {self.cid}] Using LR: {lr:.6f}")  # ? Verify LR is applied
    
        # Create Trainer for Training
        trainer = Trainer(
            max_epochs=local_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            limit_train_batches=self.model_config['train_batch_per_epoch'],
            accumulate_grad_batches=self.model_config['grad_accumulate_steps'],
            enable_model_summary=True,
            gradient_clip_val=1.0,  # ? Prevent gradient explosion
        )
    
        # Start model training
        trainer.fit(self.model, self.train_loader)
        
        # ? Clean up memory to prevent OOM errors
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
    
        # Get updated parameters
        updated_parameters = self.get_parameters(config={})
    
        # Get some metrics
        num_examples = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 1000
        metrics = {"lr": lr}
    
        return updated_parameters, num_examples, metrics

    '''
    def fit(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """ Model Training Locally """
        print("... Start Training ...")
        # Set parameters recevied from the server
        self.set_parameters(parameters)

        # Get training config
        lr = config.get("lr", self.model_config['max_lr'])
        local_epochs = config.get("local_epochs", 10)

        # Create Trainer for Training
        trainer = Trainer(
            max_epochs=local_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            limit_train_batches=self.model_config['train_batch_per_epoch'],
            accumulate_grad_batches=self.model_config['grad_accumulate_steps'],
            enable_model_summary=False, 
        )

        # Start model training
        trainer.fit(self.model, self.train_loader)

        # Get updated parameters
        updated_parameters = self.get_parameters(config={})

        # Get some metrics
        num_examples = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 1000
        metrics = {"lr": lr}

        return updated_parameters, num_examples, metrics
    '''
    def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """ Evaluate the model on the validation Data """
        # Set the parameter received from the server
        self.set_parameters(parameters)

        # Create Trainer for Evaluation
        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            limit_val_batches=1,
        )

        # Start model evaluation
        results = trainer.validate(self.model, self.val_loader)

        # Extract some metrics
        loss = results[0].get('val/loss', 0.0)
        wer = results[0].get('val/wer', 0.0)

        # Number of valiadation examples
        num_examples = len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else 100

        # Return loss and metrics
        metrics = {'wer': wer, 'val_loss': loss}

        return float(loss), num_examples, metrics


# ---------> Client function <---------

def client_fn(cid: str) -> SpeechLLMClient:
    trainloader = train_loaders[int(cid)]
    # valloader = dev_loaders[int(cid)]
    valloader = dev_loaders[int(cid) % len(dev_loaders)] 

    model = SpeechLLMLightning(**model_config)
    tokenizer = model.llm_tokenizer

    return SpeechLLMClient(cid, model, trainloader, valloader, model_config)


# ---------> Main Execution <---------

ram_memory = 16_000 * 1024 * 1024
client_resources = {"num_cpus": 1, "num_gpus": 1}
NUM_CLIENTS =25 #1582 #316 # 65 #130
NUM_ROUNDS = 200

# Start flower simulation
my_strategy = CustomStrategy(
    fraction_fit= 1.0, #0.2,
    fraction_evaluate=0.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

# Start Simultation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=my_strategy,
    ray_init_args={
        "include_dashboard": True,
        "num_cpus": 2,
        "num_gpus": 1,
        "_memory": ram_memory,
        "object_store_memory": 10 ** 9,
    },
    client_resources=client_resources
)
