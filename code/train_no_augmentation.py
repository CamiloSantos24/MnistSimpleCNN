#!/usr/bin/env python3
"""
Script de entrenamiento personalizado para ModelM7 SIN data augmentation
Entrena 20 modelos con diferentes porcentajes de datos (5% a 100%) sin transformaciones
"""

# imports -------------------------------------------------------------------------#
import sys
import os
import argparse
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from PIL import Image
from ema import EMA
from datasets import MnistDataset
from models.modelM7 import ModelM7

def calculate_metrics(predictions, targets, num_classes=10):
    """
    Calcula Precision, Recall y F1-score macro promedio
    """
    # Convertir a numpy arrays si no lo son
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Inicializar contadores
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    valid_classes = 0
    
    for class_id in range(num_classes):
        # True Positives, False Positives, False Negatives
        tp = np.sum((predictions == class_id) & (targets == class_id))
        fp = np.sum((predictions == class_id) & (targets != class_id))
        fn = np.sum((predictions != class_id) & (targets == class_id))
        
        # Calcular precision y recall para esta clase
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calcular F1-score para esta clase
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Agregar a la suma (solo si hay samples de esta clase)
        if np.sum(targets == class_id) > 0:
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            valid_classes += 1
    
    # Calcular promedios macro
    avg_precision = precision_sum / valid_classes if valid_classes > 0 else 0
    avg_recall = recall_sum / valid_classes if valid_classes > 0 else 0
    avg_f1 = f1_sum / valid_classes if valid_classes > 0 else 0
    
    return avg_precision, avg_recall, avg_f1

def create_subset_dataset(original_dataset, percentage, seed):
    """
    Crea un subset del dataset original con el porcentaje especificado
    """
    np.random.seed(seed)
    
    total_samples = len(original_dataset)
    subset_size = int(total_samples * percentage / 100)
    
    # Crear índices aleatorios para el subset
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    subset_indices = all_indices[:subset_size]
    
    # Crear subset usando Subset de PyTorch
    subset = torch.utils.data.Subset(original_dataset, subset_indices)
    
    return subset, subset_size

def run_no_augmentation(p_seed=0, p_epochs=150, p_data_percentage=100, p_logdir="temp"):
    """
    Función de entrenamiento sin data augmentation
    """
    print(f"\n" + "="*80)
    print(f"🚀 ENTRENANDO MODELO M7 SIN AUGMENTATION - {p_data_percentage}% DE DATOS - SEED {p_seed}")
    print(f"="*80)
    
    # random number generator seed ------------------------------------------------#
    SEED = p_seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # number of epochs ------------------------------------------------------------#
    NUM_EPOCHS = p_epochs
    
    # data percentage -------------------------------------------------------------#
    DATA_PERCENTAGE = p_data_percentage

    # file names ------------------------------------------------------------------#
    if not os.path.exists("../logs/%s" % p_logdir):
        os.makedirs("../logs/%s" % p_logdir)
    
    OUTPUT_FILE = str("../logs/%s/log_%02d_percent_seed_%03d_no_aug.out" % (p_logdir, DATA_PERCENTAGE, SEED))
    MODEL_FILE = str("../logs/%s/model_%02d_percent_seed_%03d_no_aug.pth" % (p_logdir, DATA_PERCENTAGE, SEED))

    # enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        exit(0)
    
    print(f"💻 Usando dispositivo: {device}")

    # NO data augmentation - solo transformación básica para tensor ---------------#
    print("🚫 SIN DATA AUGMENTATION - Solo datos originales")
    # Sin transformaciones de augmentation, solo lo necesario para el modelo
    
    # data loader -----------------------------------------------------------------#
    print(f"📁 Cargando datasets...")
    # Usar transform=None para entrenar sin augmentation
    full_train_dataset = MnistDataset(training=True, transform=None)
    test_dataset = MnistDataset(training=False, transform=None)
    
    # Crear subset de entrenamiento con el porcentaje especificado
    train_subset, subset_size = create_subset_dataset(full_train_dataset, DATA_PERCENTAGE, SEED)
    
    print(f"📊 Dataset original: {len(full_train_dataset)} imágenes")
    print(f"📊 Subset de entrenamiento ({DATA_PERCENTAGE}%): {subset_size} imágenes")
    print(f"📊 Test dataset: {len(test_dataset)} imágenes")
    print(f"🔍 Nota: NINGUNA transformación aplicada a los datos")
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=120, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    # model selection (siempre ModelM7 con kernel 7x7) ---------------------------#
    model = ModelM7().to(device)
    print(f"🧠 Modelo: ModelM7 (kernel 7x7)")
    
    try:
        summary(model, (1, 28, 28))
    except:
        print("⚠️  Resumen del modelo no disponible")

    # hyperparameter selection ----------------------------------------------------#
    ema = EMA(model, decay=0.999)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # create result file ----------------------------------------------------------#
    f = open(OUTPUT_FILE, 'w')
    f.write("# ModelM7 Training Log - NO DATA AUGMENTATION\n")
    f.write(f"# Data Percentage: {DATA_PERCENTAGE}% ({subset_size} samples)\n")
    f.write(f"# Seed: {SEED}\n")
    f.write(f"# Epochs: {NUM_EPOCHS}\n")
    f.write("# NO TRANSFORMATIONS APPLIED TO DATA\n")
    f.write("# Epoch   Train_Loss   Train_Acc   Test_Loss   Test_Acc   Best_Test_Acc   Precision   Recall   F1_Score\n")
    f.close()

    # global variables ------------------------------------------------------------#
    g_step = 0
    max_correct = 0
    
    print(f"🔥 Iniciando entrenamiento por {NUM_EPOCHS} épocas...")

    # training and evaluation loop ------------------------------------------------#
    for epoch in range(NUM_EPOCHS):
        #--------------------------------------------------------------------------#
        # train process                                                            #
        #--------------------------------------------------------------------------#
        model.train()
        train_loss = 0
        train_corr = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            train_pred = output.argmax(dim=1, keepdim=True)
            train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            loss.backward()
            optimizer.step()
            g_step += 1
            ema(model, g_step)
            
            if batch_idx % 50 == 0:  # Progreso más frecuente para datasets pequeños
                progress = 100. * batch_idx / len(train_loader)
                print(f'Época: {epoch:3d} [{batch_idx:3d}/{len(train_loader)} ({progress:.0f}%)] Loss: {loss.item():.6f}')
        
        train_loss /= len(train_subset)  # Usar el tamaño del subset
        train_accuracy = 100 * train_corr / len(train_subset)

        #--------------------------------------------------------------------------#
        # test process                                                             #
        #--------------------------------------------------------------------------#
        model.eval()
        ema.assign(model)
        test_loss = 0
        correct = 0
        total_pred = np.zeros(0)
        total_target = np.zeros(0)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device, dtype=torch.int64)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                total_pred = np.append(total_pred, pred.cpu().numpy())
                total_target = np.append(total_target, target.cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
            
            if(max_correct < correct):
                torch.save(model.state_dict(), MODEL_FILE)
                max_correct = correct
                print(f"🏆 Mejor accuracy! Imágenes correctas: {correct}/10000")
        
        ema.resume(model)

        #--------------------------------------------------------------------------#
        # output                                                                   #
        #--------------------------------------------------------------------------#
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)
        best_test_accuracy = 100 * max_correct / len(test_loader.dataset)
        
        # Calcular métricas adicionales
        precision, recall, f1_score = calculate_metrics(total_pred, total_target)
        
        print(f'📊 Época {epoch:3d}: Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.2f}%')
        print(f'                Test Loss: {test_loss:.6f}, Test Acc: {test_accuracy:.2f}% (Best: {best_test_accuracy:.2f}%)')
        print(f'                Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}')

        # Guardar en log
        f = open(OUTPUT_FILE, 'a')
        f.write(" %3d %12.6f %9.3f %12.6f %9.3f %9.3f %9.4f %9.4f %9.4f\n" % (
            epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
            best_test_accuracy, precision, recall, f1_score))
        f.close()

        #--------------------------------------------------------------------------#
        # update learning rate scheduler                                           #
        #--------------------------------------------------------------------------#
        lr_scheduler.step()
    
    print(f"✅ Entrenamiento completado!")
    print(f"🎯 Mejor accuracy: {best_test_accuracy:.3f}%")
    print(f"📁 Log guardado: {OUTPUT_FILE}")
    print(f"💾 Modelo guardado: {MODEL_FILE}")
    
    return best_test_accuracy

def main():
    """Función principal que ejecuta el entrenamiento progresivo sin augmentation"""
    print("="*80)
    print("🚫 ENTRENAMIENTO SIN DATA AUGMENTATION - MODELM7 CON DATOS PROGRESIVOS")
    print("="*80)
    
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Entrenamiento ModelM7 sin data augmentation')
    parser.add_argument("--gpu", default=0, type=int, help="GPU ID a usar (default: 0)")
    parser.add_argument("--epochs", default=150, type=int, help="Número de épocas (default: 150)")
    parser.add_argument("--logdir", default="modelM7_no_augmentation", type=str, help="Directorio de logs")
    parser.add_argument("--start_percent", default=5, type=int, help="Porcentaje inicial (default: 5)")
    parser.add_argument("--end_percent", default=100, type=int, help="Porcentaje final (default: 100)")
    parser.add_argument("--step_percent", default=5, type=int, help="Incremento de porcentaje (default: 5)")
    args = parser.parse_args()
    
    # Configurar GPU específica
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Configuración
    base_seed = 42
    
    # Lista de porcentajes configurables
    percentages = list(range(args.start_percent, args.end_percent + 1, args.step_percent))
    
    print(f"📋 Configuración:")
    print(f"   - GPU: {args.gpu}")
    print(f"   - Modelo: ModelM7 (kernel 7x7)")
    print(f"   - Épocas por modelo: {args.epochs}")
    print(f"   - Porcentajes: {percentages}")
    print(f"   - Total de modelos: {len(percentages)}")
    print(f"   - Directorio de logs: ../logs/{args.logdir}")
    print(f"   - 🚫 SIN DATA AUGMENTATION")
    
    # Almacenar resultados
    results = []
    
    for i, percentage in enumerate(percentages):
        seed = base_seed + i  # Cambiar seed para cada modelo
        
        print(f"\n{'='*60}")
        print(f"📈 PROGRESO: {i+1}/{len(percentages)} modelos")
        print(f"{'='*60}")
        
        try:
            best_accuracy = run_no_augmentation(
                p_seed=seed,
                p_epochs=args.epochs,
                p_data_percentage=percentage,
                p_logdir=args.logdir
            )
            
            results.append({
                'percentage': percentage,
                'seed': seed,
                'best_accuracy': best_accuracy
            })
            
        except Exception as e:
            print(f"❌ Error en modelo {percentage}%: {e}")
            results.append({
                'percentage': percentage,
                'seed': seed,
                'best_accuracy': 0.0
            })
    
    # Resumen final
    print(f"\n" + "="*80)
    print("📊 RESUMEN FINAL - TODOS LOS MODELOS (SIN AUGMENTATION)")
    print("="*80)
    
    print(f"{'Porcentaje':<12} {'Seed':<6} {'Best Accuracy':<14} {'Muestras':<10}")
    print("-" * 50)
    
    for result in results:
        percentage = result['percentage']
        seed = result['seed']
        accuracy = result['best_accuracy']
        samples = int(60000 * percentage / 100)
        
        print(f"{percentage:>10}% {seed:>6} {accuracy:>12.3f}% {samples:>8}")
    
    # Crear archivo resumen
    summary_file = f"../logs/{args.logdir}/training_summary_no_aug.txt"
    with open(summary_file, 'w') as f:
        f.write("# ModelM7 Progressive Training Summary - NO DATA AUGMENTATION\n")
        f.write(f"# Total models: {len(percentages)}\n")
        f.write(f"# Epochs per model: {args.epochs}\n")
        f.write("# NO TRANSFORMATIONS APPLIED\n")
        f.write("#\n")
        f.write("# Percentage  Seed  Best_Accuracy  Samples\n")
        
        for result in results:
            percentage = result['percentage']
            seed = result['seed']
            accuracy = result['best_accuracy']
            samples = int(60000 * percentage / 100)
            f.write(f"{percentage:>10}  {seed:>4}  {accuracy:>12.3f}  {samples:>7}\n")
    
    print(f"\n📁 Resumen guardado en: {summary_file}")
    print("✅ Entrenamiento progresivo sin augmentation completado!")

if __name__ == "__main__":
    main()
