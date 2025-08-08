import json
import pickle
import time
import fcntl
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class TokenizationCheckpoint:
    """Manages checkpoint information for tokenization process."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / "tokenization_checkpoint.json"
        self.completed_stages: List[str] = []
        self.current_stage: Optional[str] = None
        self.stage_files: Dict[str, List[str]] = {}
        
    def _acquire_lock(self, timeout=30):
        """Acquire a file lock to prevent race conditions in parallel processing."""
        lock_file = self.output_dir / "checkpoint.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with open(lock_file, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return f
            except (IOError, OSError):
                time.sleep(0.1)
                continue
        raise TimeoutError("Could not acquire checkpoint lock")
    
    def _release_lock(self, lock_handle):
        """Release the file lock."""
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        except:
            pass
    
    def load(self) -> bool:
        """Load checkpoint information from file. Returns True if checkpoint exists."""
        if not self.checkpoint_file.exists():
            return False
            
        lock_handle = None
        try:
            lock_handle = self._acquire_lock()
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.completed_stages = data.get('completed_stages', [])
                self.current_stage = data.get('current_stage')
                self.stage_files = data.get('stage_files', {})
                logger.info(f"Loaded checkpoint: completed stages: {self.completed_stages}")
                if self.current_stage:
                    logger.info(f"Current stage: {self.current_stage}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False
        finally:
            if lock_handle:
                self._release_lock(lock_handle)
    
    def save(self):
        """Save current checkpoint information to file."""
        data = {
            'completed_stages': self.completed_stages,
            'current_stage': self.current_stage,
            'stage_files': self.stage_files
        }
        
        lock_handle = None
        try:
            lock_handle = self._acquire_lock()
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
        finally:
            if lock_handle:
                self._release_lock(lock_handle)
    
    def mark_stage_completed(self, stage_name: str, stage_files: List[str]):
        """Mark a stage as completed and save its output files."""
        if stage_name in self.completed_stages:
            return
            
        self.completed_stages.append(stage_name)
        self.stage_files[stage_name] = stage_files
        self.current_stage = None
        self.save()
        logger.info(f"Marked stage '{stage_name}' as completed")
    
    def set_current_stage(self, stage_name: str):
        """Set the current stage being processed."""
        self.current_stage = stage_name
        self.save()
        logger.info(f"Set current stage to '{stage_name}'")
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has been completed."""
        return stage_name in self.completed_stages
    
    def get_completed_stage_files(self, stage_name: str) -> List[str]:
        """Get the output files for a completed stage."""
        return self.stage_files.get(stage_name, [])
    
    def get_last_completed_stage(self) -> Optional[str]:
        """Get the name of the last completed stage."""
        return self.completed_stages[-1] if self.completed_stages else None
    
    def clear(self):
        """Clear all checkpoint information."""
        lock_handle = None
        try:
            lock_handle = self._acquire_lock()
            self.completed_stages = []
            self.current_stage = None
            self.stage_files = {}
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            logger.info("Cleared checkpoint information")
        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {e}")
        finally:
            if lock_handle:
                self._release_lock(lock_handle)


def verify_stage_outputs(output_dir: Path, stage_name: str, expected_files: List[str]) -> bool:
    """Verify that all expected output files exist for a stage."""
    stage_dir = output_dir / stage_name
    if not stage_dir.exists():
        return False
        
    for expected_file in expected_files:
        if not (stage_dir / expected_file).exists():
            return False
    return True


def get_stage_output_files(stage_dir: Path) -> List[str]:
    """Get list of output files in a stage directory."""
    if not stage_dir.exists():
        return []
    return [f.name for f in stage_dir.iterdir() if f.is_file()]


def wait_for_checkpoint_consistency(output_dir: Path, stage_name: str, timeout=300):
    """Wait for all workers to complete a stage before proceeding."""
    checkpoint_file = output_dir / "tokenization_checkpoint.json"
    stage_dir = output_dir / stage_name
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if checkpoint file exists and stage is marked as completed
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    if stage_name in data.get('completed_stages', []):
                        logger.info(f"Stage {stage_name} marked as completed in checkpoint")
                        return True
            except:
                pass
        
        # Check if stage directory has expected files
        if stage_dir.exists() and any(stage_dir.iterdir()):
            logger.info(f"Stage {stage_name} directory has files, waiting for checkpoint update")
            time.sleep(2)
        else:
            time.sleep(1)
    
    logger.warning(f"Timeout waiting for stage {stage_name} completion")
    return False
