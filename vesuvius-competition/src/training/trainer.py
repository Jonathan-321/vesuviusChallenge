"""
Training engine with advanced features
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime


class Trainer:
    """
    Complete training engine with all bells and whistles
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = config['training'].get('gradient_accumulation', 1)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
        
        # WandB
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb:
            exp_name = config['experiment']['name']
            wandb.init(
                project=config['project']['name'],
                name=exp_name,
                config=config
            )
            wandb.watch(self.model, log='all', log_freq=100)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.config['training']['epochs']}"
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            total_loss += loss.item() * self.grad_accum_steps
            current_loss = loss.item() * self.grad_accum_steps
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to WandB
            if self.use_wandb and self.global_step % self.config['logging']['log_interval'] == 0:
                wandb.log({
                    'train/loss': current_loss,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.current_epoch,
                    'train/step': self.global_step
                })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate additional metrics
        dice_score = self.calculate_dice(outputs, targets)
        
        return avg_loss, dice_score
    
    def calculate_dice(self, pred, target, threshold=0.5):
        """Calculate Dice coefficient"""
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        return dice.item()
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        print(f"💾 Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        start_epoch = self.current_epoch
        end_epoch = start_epoch + num_epochs
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {start_epoch} → {end_epoch}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.grad_accum_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, dice_score = self.validate()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch+1}/{end_epoch}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Dice Score: {dice_score:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/dice': dice_score,
                    'epoch': epoch
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  🎯 New best validation loss!")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['logging'].get('save_interval', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', is_best)
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        if self.use_wandb:
            wandb.finish()


def get_optimizer(model, config):
    """Create optimizer from config"""
    opt_config = config['training']
    opt_name = opt_config.get('optimizer', 'adamw').lower()
    lr = opt_config['learning_rate']
    weight_decay = opt_config.get('weight_decay', 1e-4)
    
    if opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = opt_config.get('momentum', 0.9)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, config):
    """Create learning rate scheduler from config"""
    sched_name = config['training'].get('scheduler', 'cosine_warmup').lower()
    epochs = config['training']['epochs']
    
    if sched_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == 'cosine_warmup':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
    elif sched_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=epochs // 3,
            gamma=0.1
        )
    elif sched_name == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    elif sched_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")