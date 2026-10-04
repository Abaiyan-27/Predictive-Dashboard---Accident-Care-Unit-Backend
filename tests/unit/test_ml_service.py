import json
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

from app.services.ml_service import (
    predict_severity,
    predict_bundle,
    _bundle_cache
)

def test_graceful_fallback_unavailable():
    """Missing contracts give status UNAVAILABLE and the DEFER label, never a guessed zone."""
    with patch('app.services.ml_service.Path.exists', return_value=False):
        result = predict_severity()
        assert result.status == "UNAVAILABLE"
        assert "Contract is missing" in result.message
        assert result.prediction == "DEFER TO CLINICAL JUDGEMENT"
        assert result.probability is None


@patch('app.services.ml_service._read_contract')
@patch('app.services.ml_service._load_bundle_model')
def test_data_rejection_refused(mock_load_bundle, mock_read_contract):
    """Test that incomplete inputs result in a REFUSED status."""
    # Mock the contract
    mock_read_contract.return_value = ({"name": "severity_triage", "version": "1.0"}, None)
    
    # Mock the loaded bundle
    mock_model = MagicMock()
    mock_load_bundle.return_value = ({
        "raw_inputs": [
            "Age", 
            "Mode of Transport to the Hospital", 
            "Time Taken To Reach Hospital", 
            "Collision With ", 
            "Site of Injury No1"
        ],
        "features": ["feature_1", "feature_2"],
        "cat_features": [],
        "completeness_floor": 0.40,
        "model": mock_model
    }, None)

    # Call predict_severity with no data (0/5 completeness)
    result = predict_severity()

    assert result.status == "REFUSED"
    assert result.completeness == 0.0
    assert "0 of 5 triage details are available" in result.message
    
    # Call predict_severity with 2/5 completeness (0.40, which meets the floor of 0.40)
    # The actual processing will fail down the line since we're not providing real features,
    # but we just want to ensure it bypasses the REFUSED check.
    result_2 = predict_severity(age="20-29", mode_of_transport="ambulance")
    assert result_2.status != "REFUSED"
    assert result_2.completeness == 0.4


@patch('app.services.ml_service.Path.exists')
@patch('app.services.ml_service.Path.read_text')
@patch('app.services.ml_service.Path.stat')
@patch('joblib.load')
def test_joblib_memory_caching(mock_joblib_load, mock_stat, mock_read_text, mock_exists):
    """Test that joblib.load is skipped when the artifact is already loaded in memory."""
    # Clear cache before testing
    _bundle_cache.clear()
    
    mock_exists.return_value = True
    mock_read_text.return_value = json.dumps({"name": "severity_triage", "version": "1.0", "artifact": "model.joblib"})
    
    # Mock st_mtime so the caching thinks the file hasn't changed
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 123456789.0
    mock_stat.return_value = mock_stat_result
    
    # Mock the joblib model
    mock_model = MagicMock()
    # predict_proba returns a 2D array: (1, num_classes)
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_model.classes_ = ["0", "1"]
    
    mock_bundle = {
        "raw_inputs": [
            "Age", "Mode of Transport to the Hospital", "Time Taken To Reach Hospital", 
            "Collision With ", "Site of Injury No1"
        ],
        "features": ["feature_1", "feature_2"],
        "cat_features": [],
        "model": mock_model
    }
    mock_joblib_load.return_value = mock_bundle
    
    # Mock the engineering logic so we don't have to build complex dataframes
    with patch('app.services.ml_service._engineer_severity_row') as mock_engineer:
        mock_engineer.return_value = pd.DataFrame([{"feature_1": 1, "feature_2": 2}])
        
        # 1st call
        res_1 = predict_severity(
            age="20-29",
            mode_of_transport="ambulance",
            time_to_hospital="15 minutes - 30 minutes",
            collision_with="heavy vehicle",
            site_of_injury="head"
        )
        assert res_1.status == "OK"
        # joblib.load should be called exactly once
        assert mock_joblib_load.call_count == 1
        
        # 2nd call (same parameters)
        res_2 = predict_severity(
            age="20-29",
            mode_of_transport="ambulance",
            time_to_hospital="15 minutes - 30 minutes",
            collision_with="heavy vehicle",
            site_of_injury="head"
        )
        assert res_2.status == "OK"
        
        # joblib.load should STILL be 1, because it hits the _bundle_cache
        assert mock_joblib_load.call_count == 1
        
        # If we change the st_mtime (simulate a file update), the cache should invalidate
        mock_stat_result.st_mtime = 999999999.0
        
        # 3rd call
        res_3 = predict_severity(
            age="20-29",
            mode_of_transport="ambulance",
            time_to_hospital="15 minutes - 30 minutes",
            collision_with="heavy vehicle",
            site_of_injury="head"
        )
        assert res_3.status == "OK"
        
        # joblib.load should be called a second time now
        assert mock_joblib_load.call_count == 2
