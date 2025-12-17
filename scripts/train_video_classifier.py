#!/usr/bin/env python3
"""
Training script for the video classifier (Stream 1).
Trains CNN-LSTM or 3D-CNN model on the MNNIT shoplifting dataset.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_config, setup_logging, set_seed, get_device,
    save_checkpoint, load_checkpoint, count_parameters,
    plot_training_history, plot_confusion_matrix, print_classification_report
)
from datasets import ShopliftingVideoDataset, get_class_weights, collate_video_batch
from models import build_video_classifier


def load_split_data(split_file: str):
    """Load train/val/test split data from JSON file."""
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    return split_data['video_paths'], split_data['labels']


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(videos)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100.0 * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch=None):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(dataloader, desc=desc)
    
    with torch.no_grad():
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _ = model(videos)
            loss = criterion(logits, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train video classifier")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(42)
    logger = setup_logging(config['paths']['logs_dir'])
    device = get_device(config['device'])
    
    logger.info("="*60)
    logger.info("TRAINING VIDEO CLASSIFIER (STREAM 1)")
    logger.info("="*60)
    
    # Load dataset splits
    splits_dir = config['dataset']['splits_dir']
    train_paths, train_labels = load_split_data(os.path.join(splits_dir, 'train_split.json'))
    val_paths, val_labels = load_split_data(os.path.join(splits_dir, 'val_split.json'))
    test_paths, test_labels = load_split_data(os.path.join(splits_dir, 'test_split.json'))
    
    logger.info(f"Train: {len(train_paths)} videos")
    logger.info(f"Val: {len(val_paths)} videos")
    logger.info(f"Test: {len(test_paths)} videos")
    
    # Create datasets
    train_dataset = ShopliftingVideoDataset(
        video_paths=train_paths,
        labels=train_labels,
        num_frames=config['dataset']['clip_length'],
        resize=tuple(config['dataset']['input_resolution']),
        augment=True,
        config=config
    )
    
    val_dataset = ShopliftingVideoDataset(
        video_paths=val_paths,
        labels=val_labels,
        num_frames=config['dataset']['clip_length'],
        resize=tuple(config['dataset']['input_resolution']),
        augment=False,
        config=config
    )
    
    test_dataset = ShopliftingVideoDataset(
        video_paths=test_paths,
        labels=test_labels,
        num_frames=config['dataset']['clip_length'],
        resize=tuple(config['dataset']['input_resolution']),
        augment=False,
        config=config
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['video_classifier']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=collate_video_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['video_classifier']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=collate_video_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['video_classifier']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=collate_video_batch
    )
    
    # Build model
    model_config = config['models']['video_classifier']
    model_config['num_classes'] = config['dataset']['num_classes']
    model = build_video_classifier(model_config)
    model = model.to(device)
    
    logger.info(f"\nModel: {model_config['type']}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    
    # Loss function with class weights
    class_weights = get_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['video_classifier']['learning_rate'],
        weight_decay=config['training']['video_classifier']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['video_classifier']['epochs']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model, optimizer, start_epoch, _, best_val_acc = load_checkpoint(
            model, optimizer, args.resume, device
        )
        start_epoch += 1
    
    # Training loop
    logger.info("\nStarting training...")
    for epoch in range(start_epoch, config['training']['video_classifier']['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['video_classifier']['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = os.path.join(
                config['paths']['checkpoints_dir'],
                'video_classifier_best.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
            logger.info(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['video_classifier']['early_stopping_patience']:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Plot training history
    plot_path = os.path.join(config['paths']['outputs_dir'], 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    logger.info(f"\nTraining history saved to: {plot_path}")
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60)
    
    # Load best model
    best_checkpoint = os.path.join(config['paths']['checkpoints_dir'], 'video_classifier_best.pth')
    model, _, _, _, _ = load_checkpoint(model, optimizer, best_checkpoint, device)
    
    # Test
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    logger.info(f"\nTest Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Confusion matrix
    cm_path = os.path.join(config['paths']['outputs_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(
        np.array(test_labels),
        np.array(test_preds),
        class_names=['Normal', 'Shoplifting'],
        save_path=cm_path
    )
    logger.info(f"\nConfusion matrix saved to: {cm_path}")
    
    # Classification report
    report_path = os.path.join(config['paths']['outputs_dir'], 'classification_report.txt')
    print_classification_report(
        np.array(test_labels),
        np.array(test_preds),
        class_names=['Normal', 'Shoplifting'],
        save_path=report_path
    )
    logger.info(f"Classification report saved to: {report_path}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)


if __name__ == '__main__':
    main()
