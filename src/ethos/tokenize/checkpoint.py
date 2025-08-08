import json
import pickle
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
        
    def load(self) -> bool:
        """Load checkpoint information from file. Returns True if checkpoint exists."""
        if not self.checkpoint_file.exists():
            return False
            
        try:
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
    
    def save(self):
        """Save current checkpoint information to file."""
        data = {
            'completed_stages': self.completed_stages,
            'current_stage': self.current_stage,
            'stage_files': self.stage_files
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
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
        self.completed_stages = []
        self.current_stage = None
        self.stage_files = {}
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        logger.info("Cleared checkpoint information")


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
