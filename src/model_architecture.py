"""
Model Architecture Module
Universal Fiber Sensor Model with multi-head outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLayer(nn.Module):
    """Fusion layer with attention mechanism"""
    
    def __init__(self, input_dim=204, hidden_dim=256, output_dim=128, dropout=0.3):
        super(FusionLayer, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out_seq = out.unsqueeze(1)
        attn_out, _ = self.attention(out_seq, out_seq, out_seq)
        attn_out = attn_out.squeeze(1)
        
        embedding = self.fc_out(attn_out)
        return embedding


class MultiHeadClassifier(nn.Module):
    """Multi-head classifier"""
    
    def __init__(self, embedding_dim=128, num_event_classes=15, 
                 num_damage_classes=4, num_sensor_types=3):
        super(MultiHeadClassifier, self).__init__()
        
        self.event_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_event_classes)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.damage_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_damage_classes)
        )
        
        self.sensor_type_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_sensor_types)
        )
    
    def forward(self, embedding, head='all'):
        outputs = {}
        
        if head == 'all' or head == 'event':
            outputs['event_logits'] = self.event_head(embedding)
        
        if head == 'all' or head == 'risk':
            outputs['risk_score'] = self.risk_head(embedding)
        
        if head == 'all' or head == 'damage':
            outputs['damage_logits'] = self.damage_head(embedding)
        
        if head == 'all' or head == 'sensor':
            outputs['sensor_logits'] = self.sensor_type_head(embedding)
        
        return outputs


class UniversalFiberSensorModel(nn.Module):
    """Complete universal model"""
    
    def __init__(self, ufv_dim=204, embedding_dim=128, num_event_classes=15,
                 num_damage_classes=4, num_sensor_types=3):
        super(UniversalFiberSensorModel, self).__init__()
        
        self.fusion = FusionLayer(
            input_dim=ufv_dim,
            hidden_dim=256,
            output_dim=embedding_dim
        )
        
        self.classifier = MultiHeadClassifier(
            embedding_dim=embedding_dim,
            num_event_classes=num_event_classes,
            num_damage_classes=num_damage_classes,
            num_sensor_types=num_sensor_types
        )
    
    def forward(self, ufv, head='all'):
        embedding = self.fusion(ufv)
        outputs = self.classifier(embedding, head=head)
        return outputs
    
    def get_embedding(self, ufv):
        return self.fusion(ufv)
