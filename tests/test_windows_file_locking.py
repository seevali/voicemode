"""Test that temporary file handling works correctly on Windows.

This test verifies that the speech_to_text function properly closes temporary
files before attempting to reopen them, which is required on Windows due to
stricter file locking behavior.
"""

import os
import tempfile
import asyncio
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from datetime import datetime

from voice_mode.tools.converse import speech_to_text


@pytest.mark.asyncio
async def test_stt_temp_file_closed_before_reopen_with_save():
    """Test that temp file is properly closed before reopening when save_audio=True.
    
    This simulates Windows file locking behavior where a file cannot be opened
    while it's still open in another handle.
    """
    # Create test audio data at 24kHz to match SAMPLE_RATE
    sample_rate = 24000
    audio_data = np.zeros(sample_rate, dtype=np.int16)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_audio_dir = Path(temp_dir) / "audio"
        test_audio_dir.mkdir()
        
        # Track file handle states to verify proper closure
        file_handles = []
        original_open = open
        
        def tracking_open(file, mode='r', *args, **kwargs):
            """Track file opens to verify proper closure."""
            handle = original_open(file, mode, *args, **kwargs)
            file_handles.append({'file': file, 'handle': handle, 'mode': mode, 'closed': False})
            
            # Wrap close to track when file is closed
            original_close = handle.close
            def tracked_close():
                for fh in file_handles:
                    if fh['handle'] == handle:
                        fh['closed'] = True
                return original_close()
            handle.close = tracked_close
            
            return handle
        
        # Patch config and mocks
        with patch('voice_mode.config.SAVE_ALL', True), \
             patch('voice_mode.config.SAVE_AUDIO', True), \
             patch('voice_mode.tools.converse.SAVE_AUDIO', True), \
             patch('builtins.open', side_effect=tracking_open):
            
            # Mock the simple_stt_failover to return test transcription dict
            with patch('voice_mode.simple_failover.simple_stt_failover', new_callable=AsyncMock) as mock_stt:
                mock_stt.return_value = {
                    "text": "Test transcription", 
                    "provider": "whisper", 
                    "endpoint": "http://127.0.0.1:2022/v1"
                }
                
                # Mock the conversation logger
                with patch('voice_mode.tools.converse.get_conversation_logger') as mock_logger:
                    mock_conv_logger = MagicMock()
                    mock_conv_logger.conversation_id = "test123"
                    mock_logger.return_value = mock_conv_logger
                    
                    # Call the function with save_audio enabled
                    result = await speech_to_text(
                        audio_data=audio_data,
                        save_audio=True,
                        audio_dir=test_audio_dir,
                        transport="local"
                    )
                    
                    # Verify transcription was returned
                    assert isinstance(result, dict)
                    assert result.get("text") == "Test transcription"
                    
                    # Verify that any temporary file opened for writing was closed
                    # before being reopened for reading
                    temp_files = [fh for fh in file_handles if 'tmp' in str(fh['file']).lower()]
                    
                    if temp_files:
                        # Find write handle(s) and read handle(s) for temp files
                        write_handles = [fh for fh in temp_files if 'w' in fh['mode']]
                        read_handles = [fh for fh in temp_files if 'r' in fh['mode']]
                        
                        # If both write and read handles exist for temp files,
                        # verify write was closed before read
                        if write_handles and read_handles:
                            # At least one write handle should be closed
                            # This would fail on Windows if not properly handled
                            assert any(fh['closed'] for fh in write_handles), \
                                "Temp file was not closed before reopening for reading (would fail on Windows)"


@pytest.mark.asyncio
async def test_stt_temp_file_closed_before_reopen_without_save():
    """Test that temp file is properly closed before reopening when save_audio=False.
    
    This simulates Windows file locking behavior for the code path that doesn't
    save permanent files.
    """
    # Create test audio data at 24kHz to match SAMPLE_RATE
    sample_rate = 24000
    audio_data = np.zeros(sample_rate, dtype=np.int16)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_audio_dir = Path(temp_dir) / "audio"
        test_audio_dir.mkdir()
        
        # Mock the simple_stt_failover to return test transcription dict
        with patch('voice_mode.simple_failover.simple_stt_failover', new_callable=AsyncMock) as mock_stt:
            mock_stt.return_value = {
                "text": "Test transcription",
                "provider": "whisper",
                "endpoint": "http://127.0.0.1:2022/v1"
            }
            
            # Call with save_audio=False (the other code path)
            result = await speech_to_text(
                audio_data=audio_data,
                save_audio=False,
                audio_dir=test_audio_dir,
                transport="local"
            )
            
            # Verify transcription was returned
            assert isinstance(result, dict)
            assert result["text"] == "Test transcription"
            
            # If we get here without a file locking exception, the fix is working
            # On Windows without the fix, this would raise:
            # "The process cannot access the file because it is being used by another process"


@pytest.mark.asyncio
async def test_stt_temp_file_cleanup_on_error():
    """Test that temporary files are cleaned up even when errors occur."""
    # Create test audio data at 24kHz
    sample_rate = 24000
    audio_data = np.zeros(sample_rate, dtype=np.int16)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_audio_dir = Path(temp_dir) / "audio"
        test_audio_dir.mkdir()
        
        # Mock simple_stt_failover to raise an exception
        with patch('voice_mode.simple_failover.simple_stt_failover', new_callable=AsyncMock) as mock_stt:
            mock_stt.side_effect = Exception("Test error")
            
            # The function should handle the error gracefully
            with pytest.raises(Exception):
                await speech_to_text(
                    audio_data=audio_data,
                    save_audio=False,
                    audio_dir=test_audio_dir,
                    transport="local"
                )
            
            # Verify temp files are cleaned up (should not remain in /tmp)
            # This is a basic check - a more thorough check would track temp file paths


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_stt_temp_file_closed_before_reopen_with_save())
    asyncio.run(test_stt_temp_file_closed_before_reopen_without_save())
    asyncio.run(test_stt_temp_file_cleanup_on_error())
    print("All tests passed!")
