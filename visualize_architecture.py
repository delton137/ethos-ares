#!/usr/bin/env python3
"""
ETHOS Transformer Architecture Visualization
Creates publication-quality figures showing the model architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

# Set up publication-quality plotting
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def create_architecture_diagram():
    """Create the main architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E8F4FD',
        'embedding': '#B3D9FF',
        'attention': '#FFB366',
        'mlp': '#FF9999',
        'output': '#99FF99',
        'norm': '#E6E6FA',
        'text': '#333333'
    }
    
    # Title
    ax.text(5, 11.5, 'ETHOS Medical Transformer Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input section
    ax.add_patch(FancyBboxPatch((0.5, 9.5), 9, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor='black', linewidth=2))
    ax.text(5, 10.2, 'Medical Event Sequence', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 9.8, 'CONDITION//72711, LAB//3036277, DRUG//430193006, ...', 
            fontsize=10, ha='center', style='italic')
    
    # Embedding section
    ax.add_patch(FancyBboxPatch((0.5, 8), 9, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['embedding'], 
                                edgecolor='black', linewidth=2))
    ax.text(5, 8.6, 'Token + Position Embeddings', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 8.3, 'Vocabulary Size: ~25K-40K SNOMED CT Codes', 
            fontsize=10, ha='center')
    
    # Transformer blocks
    block_height = 0.8
    block_width = 1.8
    n_blocks = 6
    start_x = 1.5
    
    for i in range(n_blocks):
        x = start_x + i * 1.2
        y = 6.5
        
        # Block background
        ax.add_patch(FancyBboxPatch((x, y), block_width, block_height, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='white', 
                                    edgecolor='black', linewidth=1.5))
        
        # Block label
        ax.text(x + block_width/2, y + block_height + 0.1, f'Block {i+1}', 
                fontsize=10, fontweight='bold', ha='center')
        
        # Layer Norm 1
        ax.add_patch(Rectangle((x + 0.1, y + 0.6), block_width - 0.2, 0.15, 
                              facecolor=colors['norm'], edgecolor='black'))
        ax.text(x + block_width/2, y + 0.675, 'Layer Norm', fontsize=8, ha='center')
        
        # Multi-Head Attention
        ax.add_patch(Rectangle((x + 0.1, y + 0.4), block_width - 0.2, 0.15, 
                              facecolor=colors['attention'], edgecolor='black'))
        ax.text(x + block_width/2, y + 0.475, 'Multi-Head\nAttention', fontsize=8, ha='center')
        
        # Layer Norm 2
        ax.add_patch(Rectangle((x + 0.1, y + 0.2), block_width - 0.2, 0.15, 
                              facecolor=colors['norm'], edgecolor='black'))
        ax.text(x + block_width/2, y + 0.275, 'Layer Norm', fontsize=8, ha='center')
        
        # MLP
        ax.add_patch(Rectangle((x + 0.1, y + 0.05), block_width - 0.2, 0.15, 
                              facecolor=colors['mlp'], edgecolor='black'))
        ax.text(x + block_width/2, y + 0.125, 'MLP', fontsize=8, ha='center')
    
    # Arrows between blocks
    for i in range(n_blocks - 1):
        x1 = start_x + i * 1.2 + block_width
        x2 = start_x + (i + 1) * 1.2
        y = 6.5 + block_height/2
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Output section
    ax.add_patch(FancyBboxPatch((0.5, 5), 9, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='black', linewidth=2))
    ax.text(5, 5.6, 'Language Model Head', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 5.3, 'Next Token Prediction', fontsize=10, ha='center')
    
    # Architecture details
    details_y = 3.5
    ax.text(5, details_y + 0.5, 'Architecture Specifications', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Left column
    ax.text(2, details_y, 'Model Type: Decoder-only (GPT-2 style)', fontsize=11, ha='left')
    ax.text(2, details_y - 0.3, 'Context Window: 1,024 tokens', fontsize=11, ha='left')
    ax.text(2, details_y - 0.6, 'Layers: 6 transformer blocks', fontsize=11, ha='left')
    ax.text(2, details_y - 0.9, 'Attention Heads: 8', fontsize=11, ha='left')
    
    # Right column
    ax.text(6, details_y, 'Embedding Dim: 256', fontsize=11, ha='left')
    ax.text(6, details_y - 0.3, 'Vocabulary: ~25K-40K SNOMED CT codes', fontsize=11, ha='left')
    ax.text(6, details_y - 0.6, 'Activation: GELU', fontsize=11, ha='left')
    ax.text(6, details_y - 0.9, 'Parameters: ~2.5M', fontsize=11, ha='left')
    
    # Data flow arrows
    # Input to embedding
    ax.annotate('', xy=(5, 8), xytext=(5, 9.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Embedding to first block
    ax.annotate('', xy=(start_x, 6.5 + block_height), xytext=(5, 8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Last block to output
    ax.annotate('', xy=(5, 5), xytext=(start_x + (n_blocks-1) * 1.2 + block_width/2, 6.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    return fig

def create_attention_mechanism_diagram():
    """Create a detailed attention mechanism diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Multi-Head Self-Attention Mechanism', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input tokens
    tokens = ['CONDITION//72711', 'LAB//3036277', 'DRUG//430193006']
    token_y = 6.5
    
    for i, token in enumerate(tokens):
        x = 1 + i * 2.5
        ax.add_patch(FancyBboxPatch((x, token_y), 2, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#E8F4FD', 
                                    edgecolor='black', linewidth=1.5))
        ax.text(x + 1, token_y + 0.3, token, fontsize=10, ha='center', fontweight='bold')
    
    # Embeddings
    embed_y = 5.5
    for i in range(3):
        x = 1 + i * 2.5
        ax.add_patch(FancyBboxPatch((x, embed_y), 2, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#B3D9FF', 
                                    edgecolor='black', linewidth=1.5))
        ax.text(x + 1, embed_y + 0.3, f'E{i+1}', fontsize=12, ha='center', fontweight='bold')
    
    # Q, K, V projections
    qkv_y = 4.5
    colors = ['#FFB366', '#FF9999', '#99FF99']
    labels = ['Q', 'K', 'V']
    
    for i in range(3):
        x = 1 + i * 2.5
        for j in range(3):
            ax.add_patch(FancyBboxPatch((x + j * 0.6, qkv_y), 0.5, 0.4, 
                                        boxstyle="round,pad=0.02", 
                                        facecolor=colors[j], 
                                        edgecolor='black', linewidth=1))
            ax.text(x + j * 0.6 + 0.25, qkv_y + 0.2, f'{labels[j]}{i+1}', 
                    fontsize=8, ha='center', fontweight='bold')
    
    # Attention computation
    attn_y = 3.5
    ax.add_patch(FancyBboxPatch((2, attn_y), 6, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#FFE6CC', 
                                edgecolor='black', linewidth=1.5))
    ax.text(5, attn_y + 0.4, 'Attention(Q, K, V) = softmax(QK^T/√d_k)V', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Output
    output_y = 2.5
    ax.add_patch(FancyBboxPatch((2, output_y), 6, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#E6F3FF', 
                                edgecolor='black', linewidth=1.5))
    ax.text(5, output_y + 0.3, 'Contextualized Representations', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Arrows
    # Tokens to embeddings
    for i in range(3):
        x = 2 + i * 2.5
        ax.annotate('', xy=(x + 1, embed_y + 0.6), xytext=(x + 1, token_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Embeddings to QKV
    for i in range(3):
        x = 2 + i * 2.5
        ax.annotate('', xy=(x + 1, qkv_y + 0.4), xytext=(x + 1, embed_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # QKV to attention
    ax.annotate('', xy=(5, attn_y + 0.8), xytext=(5, qkv_y),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Attention to output
    ax.annotate('', xy=(5, output_y + 0.6), xytext=(5, attn_y),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Labels
    ax.text(0.5, 6.8, 'Input\nTokens', fontsize=10, ha='center', fontweight='bold')
    ax.text(0.5, 5.8, 'Token\nEmbeddings', fontsize=10, ha='center', fontweight='bold')
    ax.text(0.5, 4.7, 'Q, K, V\nProjections', fontsize=10, ha='center', fontweight='bold')
    ax.text(0.5, 3.9, 'Attention\nComputation', fontsize=10, ha='center', fontweight='bold')
    ax.text(0.5, 2.8, 'Output\nRepresentations', fontsize=10, ha='center', fontweight='bold')
    
    return fig

def create_data_flow_diagram():
    """Create a data flow diagram showing the complete pipeline"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'ETHOS Medical AI Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Pipeline stages
    stages = [
        ('All of Us\nOMOP Data', 1, 7.5, '#E8F4FD'),
        ('SNOMED CT\nConversion', 3.5, 7.5, '#B3D9FF'),
        ('MEDS\nFormat', 6, 7.5, '#FFB366'),
        ('ETHOS\nTokenization', 8.5, 7.5, '#FF9999'),
        ('Transformer\nTraining', 11, 7.5, '#99FF99'),
        ('Medical AI\nModel', 13.5, 7.5, '#E6E6FA')
    ]
    
    # Draw stages
    for name, x, y, color in stages:
        ax.add_patch(FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, 
                                    edgecolor='black', linewidth=2))
        ax.text(x, y, name, fontsize=11, ha='center', fontweight='bold')
    
    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 0.8
        x2 = stages[i+1][1] - 0.8
        y = stages[i][2]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Data examples
    examples = [
        ('condition_concept_id: 72711\nperson_id: 12345\ntime: 2020-01-15', 1, 5.5),
        ('CONDITION//72711\nLAB//3036277\nDRUG//430193006', 3.5, 5.5),
        ('subject_id: 12345.0\ntime: 1579094400000000\ncode: CONDITION//72711', 6, 5.5),
        ('token_ids: [1024, 2048, 3072]\nvocab_size: 25,000', 8.5, 5.5),
        ('loss: 2.34\naccuracy: 0.85\nparams: 2.5M', 11, 5.5),
        ('Medical predictions\nRisk assessment\nTreatment recommendations', 13.5, 5.5)
    ]
    
    for text, x, y in examples:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.8), 2.4, 1.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='white', 
                                    edgecolor='gray', linewidth=1, linestyle='--'))
        ax.text(x, y, text, fontsize=9, ha='center', va='center')
    
    # Key features
    features_y = 3
    ax.text(8, features_y + 0.5, 'Key Features', 
            fontsize=16, fontweight='bold', ha='center')
    
    features = [
        '• Standardized SNOMED CT vocabulary',
        '• 7 OMOP tables (conditions, drugs, labs, etc.)',
        '• Causal self-attention for temporal modeling',
        '• 6-layer transformer with 8 attention heads',
        '• ~25K-40K medical concept vocabulary',
        '• End-to-end medical AI training'
    ]
    
    for i, feature in enumerate(features):
        y = features_y - i * 0.4
        ax.text(1, y, feature, fontsize=11, ha='left')
    
    return fig

def main():
    """Generate all architecture diagrams"""
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating ETHOS architecture diagrams...")
    
    # Main architecture diagram
    fig1 = create_architecture_diagram()
    fig1.savefig(output_dir / 'ethos_architecture.png', bbox_inches='tight', dpi=300)
    fig1.savefig(output_dir / 'ethos_architecture.pdf', bbox_inches='tight')
    plt.close(fig1)
    print("✓ Main architecture diagram saved")
    
    # Attention mechanism diagram
    fig2 = create_attention_mechanism_diagram()
    fig2.savefig(output_dir / 'attention_mechanism.png', bbox_inches='tight', dpi=300)
    fig2.savefig(output_dir / 'attention_mechanism.pdf', bbox_inches='tight')
    plt.close(fig2)
    print("✓ Attention mechanism diagram saved")
    
    # Data flow diagram
    fig3 = create_data_flow_diagram()
    fig3.savefig(output_dir / 'data_flow_pipeline.png', bbox_inches='tight', dpi=300)
    fig3.savefig(output_dir / 'data_flow_pipeline.pdf', bbox_inches='tight')
    plt.close(fig3)
    print("✓ Data flow pipeline diagram saved")
    
    print(f"\nAll diagrams saved to {output_dir}/")
    print("Files generated:")
    print("  - ethos_architecture.png/pdf")
    print("  - attention_mechanism.png/pdf") 
    print("  - data_flow_pipeline.png/pdf")

if __name__ == "__main__":
    main()
