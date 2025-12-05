"""
Test Suite for Model Architecture Module

Tests FusionLayer, MultiHeadClassifier, and UniversalFiberSensorModel.

This test suite validates the neural network architecture according to:
- IEEE standards for neural network validation
- PyTorch best practices for model testing

Test categories:
1. FusionLayer tests
2. MultiHeadClassifier tests
3. UniversalFiberSensorModel tests
4. Gradient flow tests
5. Batch processing tests

Coverage target: 75%+
Test count: 15 tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.model_architecture import (
    FusionLayer,
    MultiHeadClassifier,
    UniversalFiberSensorModel
)


class TestFusionLayer:
    """
    Tests for FusionLayer with attention mechanism.
    
    Input: 204-dim UFV
    Output: 128-dim embedding
    """
    
    @pytest.mark.unit
    def test_fusion_input_dimension(self, fusion_layer, sample_ufv):
        """
        Test FusionLayer accepts 204-dimensional input.
        """
        # Add batch dimension
        input_tensor = sample_ufv.unsqueeze(0)
        
        # Should not raise any errors
        output = fusion_layer(input_tensor)
        
        assert output is not None
    
    @pytest.mark.unit
    def test_fusion_output_dimension(self, fusion_layer, sample_ufv, expected_dimensions):
        """
        Test FusionLayer outputs 128-dimensional embedding.
        """
        input_tensor = sample_ufv.unsqueeze(0)
        output = fusion_layer(input_tensor)
        
        assert output.shape == (1, expected_dimensions['embedding'])
    
    @pytest.mark.unit
    def test_fusion_batch_processing(self, fusion_layer, batch_ufv, expected_dimensions):
        """
        Test FusionLayer handles batch sizes > 1.
        """
        output = fusion_layer(batch_ufv)
        
        assert output.shape == (8, expected_dimensions['embedding'])
    
    @pytest.mark.unit
    def test_fusion_no_nan_output(self, fusion_layer, sample_ufv):
        """
        Test FusionLayer output contains no NaN values.
        """
        input_tensor = sample_ufv.unsqueeze(0)
        output = fusion_layer(input_tensor)
        
        assert not torch.any(torch.isnan(output))
    
    @pytest.mark.unit
    def test_fusion_attention_mechanism(self, sample_ufv):
        """
        Test FusionLayer attention produces valid outputs.
        
        The attention mechanism should properly weight the features.
        """
        fusion = FusionLayer(input_dim=204, hidden_dim=256, output_dim=128)
        fusion.eval()
        
        input_tensor = sample_ufv.unsqueeze(0)
        
        with torch.no_grad():
            output = fusion(input_tensor)
        
        # Output should be bounded (typical activation range)
        assert torch.all(torch.abs(output) < 100), "Output values unexpectedly large"


class TestMultiHeadClassifier:
    """
    Tests for MultiHeadClassifier with 4 heads.
    
    Heads: event (15), risk (1), damage (4), sensor (3)
    """
    
    @pytest.mark.unit
    def test_event_head_output(self, multi_head_classifier, sample_embedding, expected_dimensions):
        """
        Test event head outputs 15 class logits.
        """
        outputs = multi_head_classifier(sample_embedding, head='event')
        
        assert 'event_logits' in outputs
        assert outputs['event_logits'].shape == (1, expected_dimensions['event_classes'])
    
    @pytest.mark.unit
    def test_risk_head_output(self, multi_head_classifier, sample_embedding):
        """
        Test risk head outputs sigmoid value in [0, 1].
        """
        outputs = multi_head_classifier(sample_embedding, head='risk')
        
        assert 'risk_score' in outputs
        assert outputs['risk_score'].shape == (1, 1)
        
        # Risk score should be in [0, 1] due to sigmoid
        risk = outputs['risk_score'].item()
        assert 0 <= risk <= 1, f"Risk {risk} should be in [0, 1]"
    
    @pytest.mark.unit
    def test_damage_head_output(self, multi_head_classifier, sample_embedding, expected_dimensions):
        """
        Test damage head outputs 4 class logits.
        """
        outputs = multi_head_classifier(sample_embedding, head='damage')
        
        assert 'damage_logits' in outputs
        assert outputs['damage_logits'].shape == (1, expected_dimensions['damage_classes'])
    
    @pytest.mark.unit
    def test_sensor_head_output(self, multi_head_classifier, sample_embedding, expected_dimensions):
        """
        Test sensor type head outputs 3 class logits.
        """
        outputs = multi_head_classifier(sample_embedding, head='sensor')
        
        assert 'sensor_logits' in outputs
        assert outputs['sensor_logits'].shape == (1, expected_dimensions['sensor_types'])
    
    @pytest.mark.unit
    def test_head_selection_all(self, multi_head_classifier, sample_embedding):
        """
        Test head='all' returns all 4 outputs.
        """
        outputs = multi_head_classifier(sample_embedding, head='all')
        
        assert 'event_logits' in outputs
        assert 'risk_score' in outputs
        assert 'damage_logits' in outputs
        assert 'sensor_logits' in outputs
    
    @pytest.mark.unit
    def test_head_selection_single(self, multi_head_classifier, sample_embedding):
        """
        Test single head selection returns only that head.
        """
        outputs = multi_head_classifier(sample_embedding, head='event')
        
        assert 'event_logits' in outputs
        # Other heads should not be computed
        assert 'risk_score' not in outputs or len(outputs) == 1


class TestUniversalFiberSensorModel:
    """
    Tests for the complete UniversalFiberSensorModel.
    
    Full pipeline: UFV (204) -> Embedding (128) -> Predictions
    """
    
    @pytest.mark.unit
    def test_model_forward_pass(self, model, sample_ufv):
        """
        Test complete forward pass works without errors.
        """
        input_tensor = sample_ufv.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor, head='all')
        
        assert outputs is not None
        assert len(outputs) == 4  # 4 heads
    
    @pytest.mark.unit
    def test_model_output_structure(self, model, sample_ufv):
        """
        Test model output contains all expected keys.
        """
        input_tensor = sample_ufv.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor, head='all')
        
        expected_keys = ['event_logits', 'risk_score', 'damage_logits', 'sensor_logits']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
    
    @pytest.mark.unit
    def test_model_gradient_flow(self, model_training_mode, sample_ufv):
        """
        Test gradients propagate through model in training mode.
        """
        input_tensor = sample_ufv.unsqueeze(0)
        input_tensor.requires_grad = True
        
        outputs = model_training_mode(input_tensor, head='all')
        
        # Sum all outputs and compute gradient
        loss = outputs['event_logits'].sum() + outputs['risk_score'].sum()
        loss.backward()
        
        # Input should have gradient
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0), "Gradient should not be all zeros"
    
    @pytest.mark.unit
    def test_get_embedding(self, model, sample_ufv, expected_dimensions):
        """
        Test get_embedding returns 128-dimensional embedding.
        """
        input_tensor = sample_ufv.unsqueeze(0)
        
        with torch.no_grad():
            embedding = model.get_embedding(input_tensor)
        
        assert embedding.shape == (1, expected_dimensions['embedding'])
    
    @pytest.mark.unit
    def test_model_deterministic(self, model, sample_ufv):
        """
        Test model produces identical output for same input in eval mode.
        """
        model.eval()
        input_tensor = sample_ufv.unsqueeze(0)
        
        with torch.no_grad():
            output1 = model(input_tensor, head='all')
            output2 = model(input_tensor, head='all')
        
        # Event logits should be identical
        torch.testing.assert_close(
            output1['event_logits'],
            output2['event_logits'],
            msg="Model should be deterministic in eval mode"
        )
    
    @pytest.mark.unit
    def test_model_batch_processing(self, model, batch_ufv, expected_dimensions):
        """
        Test model handles batch input correctly.
        """
        with torch.no_grad():
            outputs = model(batch_ufv, head='all')
        
        # Check batch dimension preserved
        assert outputs['event_logits'].shape[0] == 8
        assert outputs['risk_score'].shape[0] == 8
        assert outputs['damage_logits'].shape[0] == 8
        assert outputs['sensor_logits'].shape[0] == 8
        
        # Check output dimensions
        assert outputs['event_logits'].shape[1] == expected_dimensions['event_classes']
        assert outputs['damage_logits'].shape[1] == expected_dimensions['damage_classes']
        assert outputs['sensor_logits'].shape[1] == expected_dimensions['sensor_types']

