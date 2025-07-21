#!/usr/bin/env python3
"""
Script para graficar los resultados de entrenamiento progresivo
Compara modelos con y sin data augmentation
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def read_summary_file(file_path):
    """
    Lee un archivo de resumen y extrae los datos de porcentaje y accuracy
    """
    percentages = []
    accuracies = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Buscar líneas de datos (que no empiecen con #)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    percentage = int(parts[0])
                    accuracy = float(parts[2])
                    percentages.append(percentage)
                    accuracies.append(accuracy)
        
        return percentages, accuracies
    
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {file_path}")
        return [], []
    except Exception as e:
        print(f"❌ Error al leer {file_path}: {e}")
        return [], []

def plot_comparison_graphs():
    """
    Crea dos gráficas separadas para comparar resultados con y sin augmentation
    """
    # Rutas de los archivos
    aug_file = "../logs/modelM7_progressive/training_summary.txt"
    no_aug_file = "../logs/modelM7_no_augmentation/training_summary_no_aug.txt"
    
    # Leer datos
    print("📊 Leyendo datos de entrenamiento...")
    percentages_aug, accuracies_aug = read_summary_file(aug_file)
    percentages_no_aug, accuracies_no_aug = read_summary_file(no_aug_file)
    
    if not percentages_aug and not percentages_no_aug:
        print("❌ No se pudieron leer los datos de ningún archivo")
        return
    
    # Configurar el estilo de las gráficas
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colores
    color_aug = '#2E86AB'      # Azul para con augmentation
    color_no_aug = '#A23B72'   # Magenta para sin augmentation
    
    #--------------------------------------------------------------------------#
    # Gráfica 1: Con Data Augmentation
    #--------------------------------------------------------------------------#
    if percentages_aug:
        ax1.plot(percentages_aug, accuracies_aug, 
                marker='o', linewidth=2.5, markersize=6, 
                color=color_aug, markerfacecolor='white', 
                markeredgecolor=color_aug, markeredgewidth=2)
        
        ax1.set_title('ModelM7 - CON Data Augmentation', fontsize=14, fontweight='bold', color=color_aug)
        ax1.set_xlabel('Porcentaje de Datos de Entrenamiento (%)', fontsize=12)
        ax1.set_ylabel('Mejor Accuracy (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 105)
        
        # Configurar ticks del eje y (0 a 100)
        ax1.set_ylim(0, 105)
        
        # Añadir valores en puntos clave
        for i, (x, y) in enumerate(zip(percentages_aug, accuracies_aug)):
            if x in [5, 25, 50, 75, 100]:  # Mostrar algunos valores clave
                ax1.annotate(f'{y:.2f}%', (x, y), 
                           textcoords="offset points", xytext=(0,10), ha='center', 
                           fontsize=9, color=color_aug, fontweight='bold')
        
        print(f"✅ Gráfica 1: Con augmentation - {len(percentages_aug)} puntos")
        print(f"   Rango accuracy: {min(accuracies_aug):.3f}% - {max(accuracies_aug):.3f}%")
    else:
        ax1.text(0.5, 0.5, 'Datos no disponibles\nCon Data Augmentation', 
                ha='center', va='center', transform=ax1.transAxes, 
                fontsize=12, color='red')
        ax1.set_title('ModelM7 - CON Data Augmentation', fontsize=14, fontweight='bold')
    
    #--------------------------------------------------------------------------#
    # Gráfica 2: Sin Data Augmentation
    #--------------------------------------------------------------------------#
    if percentages_no_aug:
        ax2.plot(percentages_no_aug, accuracies_no_aug, 
                marker='s', linewidth=2.5, markersize=6, 
                color=color_no_aug, markerfacecolor='white', 
                markeredgecolor=color_no_aug, markeredgewidth=2)
        
        ax2.set_title('ModelM7 - SIN Data Augmentation', fontsize=14, fontweight='bold', color=color_no_aug)
        ax2.set_xlabel('Porcentaje de Datos de Entrenamiento (%)', fontsize=12)
        ax2.set_ylabel('Mejor Accuracy (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 105)
        
        # Configurar ticks del eje y (0 a 100)
        ax2.set_ylim(0, 105)
        
        # Añadir valores en puntos clave
        for i, (x, y) in enumerate(zip(percentages_no_aug, accuracies_no_aug)):
            if x in [5, 25, 50, 75, 100]:  # Mostrar algunos valores clave
                ax2.annotate(f'{y:.2f}%', (x, y), 
                           textcoords="offset points", xytext=(0,10), ha='center', 
                           fontsize=9, color=color_no_aug, fontweight='bold')
        
        print(f"✅ Gráfica 2: Sin augmentation - {len(percentages_no_aug)} puntos")
        print(f"   Rango accuracy: {min(accuracies_no_aug):.3f}% - {max(accuracies_no_aug):.3f}%")
    else:
        ax2.text(0.5, 0.5, 'Datos no disponibles\nSin Data Augmentation', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=12, color='red')
        ax2.set_title('ModelM7 - SIN Data Augmentation', fontsize=14, fontweight='bold')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfica
    output_file = "../logs/comparison_accuracy_vs_data_percentage.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfica guardada: {output_file}")
    
    # Mostrar estadísticas comparativas
    if percentages_aug and percentages_no_aug:
        print(f"\n📈 ESTADÍSTICAS COMPARATIVAS:")
        print(f"{'Métrica':<25} {'Con Augmentation':<18} {'Sin Augmentation':<18}")
        print("-" * 65)
        print(f"{'Accuracy Promedio':<25} {np.mean(accuracies_aug):>15.3f}% {np.mean(accuracies_no_aug):>15.3f}%")
        print(f"{'Accuracy Máxima':<25} {np.max(accuracies_aug):>15.3f}% {np.max(accuracies_no_aug):>15.3f}%")
        print(f"{'Accuracy Mínima':<25} {np.min(accuracies_aug):>15.3f}% {np.min(accuracies_no_aug):>15.3f}%")
        print(f"{'Desviación Estándar':<25} {np.std(accuracies_aug):>15.3f}% {np.std(accuracies_no_aug):>15.3f}%")
        
        # Calcular diferencias
        if len(accuracies_aug) == len(accuracies_no_aug):
            differences = np.array(accuracies_aug) - np.array(accuracies_no_aug)
            print(f"{'Diferencia Promedio':<25} {np.mean(differences):>15.3f}% {'(Aug - No Aug)':<15}")
    
    # Mostrar la gráfica
    plt.show()

def plot_combined_comparison():
    """
    Crea una gráfica combinada para comparar ambos entrenamientos
    """
    # Rutas de los archivos
    aug_file = "../logs/modelM7_progressive/training_summary.txt"
    no_aug_file = "../logs/modelM7_no_augmentation/training_summary_no_aug.txt"
    
    # Leer datos
    percentages_aug, accuracies_aug = read_summary_file(aug_file)
    percentages_no_aug, accuracies_no_aug = read_summary_file(no_aug_file)
    
    if not percentages_aug or not percentages_no_aug:
        print("⚠️  No hay suficientes datos para la comparación combinada")
        return
    
    # Crear gráfica combinada
    plt.figure(figsize=(12, 8))
    
    # Colores
    color_aug = '#2E86AB'      # Azul para con augmentation
    color_no_aug = '#A23B72'   # Magenta para sin augmentation
    
    # Plotear ambas líneas
    plt.plot(percentages_aug, accuracies_aug, 
             marker='o', linewidth=3, markersize=8, 
             color=color_aug, markerfacecolor='white', 
             markeredgecolor=color_aug, markeredgewidth=2,
             label='Con Data Augmentation', alpha=0.8)
    
    plt.plot(percentages_no_aug, accuracies_no_aug, 
             marker='s', linewidth=3, markersize=8, 
             color=color_no_aug, markerfacecolor='white', 
             markeredgecolor=color_no_aug, markeredgewidth=2,
             label='Sin Data Augmentation', alpha=0.8)
    
    # Configuración de la gráfica
    plt.title('Comparación ModelM7: Con vs Sin Data Augmentation', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Porcentaje de Datos de Entrenamiento (%)', fontsize=14)
    plt.ylabel('Mejor Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 105)
    
    # Configurar rango del eje y (0 a 100)
    plt.ylim(0, 105)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfica combinada
    output_file = "../logs/combined_accuracy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfica combinada guardada: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("📊 GRAFICANDO RESULTADOS DE ENTRENAMIENTO PROGRESIVO")
    print("="*70)
    
    # Cambiar al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n🎯 Creando gráficas separadas...")
    plot_comparison_graphs()
    
    print("\n🎯 Creando gráfica combinada...")
    plot_combined_comparison()
    
    print("\n✅ Graficación completada!")
