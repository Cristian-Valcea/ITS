# src/api/services.py
import yaml
import os
import logging
from typing import Dict, Any, Union

# Assuming Pydantic models from config_models.py are used for validation if needed
# from .config_models import MainConfig, ModelParamsConfig, RiskLimitsConfig # Uncomment when ready

logger = logging.getLogger("RLTradingPlatform.APIServices")

class ConfigService:
    """
    Service class for managing (reading and writing) YAML configuration files.
    """
    def __init__(self, main_config_path: str, model_params_path: str, risk_limits_path: str):
        self.config_paths = {
            "main_config": main_config_path,
            "model_params": model_params_path,
            "risk_limits": risk_limits_path,
        }
        self.logger = logging.getLogger(f"{__name__}.ConfigService")
        self.logger.info(f"ConfigService initialized with paths: {self.config_paths}")
        self._ensure_config_files_exist()


    def _ensure_config_files_exist(self):
        """Creates empty YAML files if they don't exist, to prevent errors on first load."""
        for name, path in self.config_paths.items():
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    yaml.dump({}, f) # Create an empty YAML structure
                self.logger.info(f"Created empty default config file: {path} for '{name}'")


    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Reads a specified YAML configuration file.

        Args:
            config_name (str): The name of the configuration to read 
                               (e.g., "main_config", "model_params", "risk_limits").

        Returns:
            Dict[str, Any]: The content of the YAML file as a dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config_name is invalid or YAML parsing fails.
        """
        if config_name not in self.config_paths:
            self.logger.error(f"Invalid config name: {config_name}")
            raise ValueError(f"Invalid configuration name: {config_name}. Valid names are: {list(self.config_paths.keys())}")

        file_path = self.config_paths[config_name]
        self.logger.debug(f"Attempting to read config: {config_name} from {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Configuration file not found: {file_path}")
            # self._ensure_config_files_exist() # Ensure it exists, then try again (might be too aggressive)
            # For get, it should ideally exist or be explicitly created by user/setup.
            # Let's assume if _ensure_config_files_exist ran at init, this shouldn't be an issue unless deleted.
            raise FileNotFoundError(f"Configuration file {file_path} for '{config_name}' not found.")

        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if config_data is None: # Handles empty file case
                config_data = {}
            self.logger.info(f"Successfully read config: {config_name} from {file_path}")
            return config_data
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML file {file_path} for {config_name}: {e}", exc_info=True)
            raise ValueError(f"Error parsing YAML for {config_name}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error reading config {config_name} from {file_path}: {e}", exc_info=True)
            raise ValueError(f"Unexpected error reading config {config_name}: {e}")


    def update_config(self, config_name: str, new_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates a specified YAML configuration file with new data.
        The new_config_data should ideally be validated against a Pydantic model before calling this.

        Args:
            config_name (str): The name of the configuration to update.
            new_config_data (Dict[str, Any]): The new configuration data.

        Returns:
            Dict[str, Any]: The updated configuration data that was written.

        Raises:
            FileNotFoundError: If the config file path is invalid (directory doesn't exist).
            ValueError: If the config_name is invalid or YAML writing fails.
        """
        if config_name not in self.config_paths:
            self.logger.error(f"Invalid config name for update: {config_name}")
            raise ValueError(f"Invalid configuration name: {config_name}. Valid names are: {list(self.config_paths.keys())}")

        file_path = self.config_paths[config_name]
        self.logger.debug(f"Attempting to update config: {config_name} at {file_path}")

        try:
            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                yaml.dump(new_config_data, f, sort_keys=False, default_flow_style=False)
            self.logger.info(f"Successfully updated config: {config_name} at {file_path}")
            return new_config_data
        except IOError as e:
            self.logger.error(f"IOError writing config {config_name} to {file_path}: {e}", exc_info=True)
            raise ValueError(f"IOError writing config {config_name}: {e}")
        except yaml.YAMLError as e: # Should not happen with dump, but good practice
            self.logger.error(f"YAMLError writing config {config_name} to {file_path}: {e}", exc_info=True)
            raise ValueError(f"YAMLError writing config {config_name}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error writing config {config_name} to {file_path}: {e}", exc_info=True)
            raise ValueError(f"Unexpected error writing config {config_name}: {e}")


# OrchestratorService could be added here later if direct method calls to OrchestratorAgent
# are preferred over subprocess calls to src/main.py for pipeline execution.
# For Phase 1, OrchestratorAgent might be instantiated and used directly in api/main.py's endpoints.

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger_main = logging.getLogger(__name__)

    # --- Example Usage for ConfigService ---
    # Create dummy config paths for testing
    test_config_dir = "temp_test_config_dir"
    os.makedirs(test_config_dir, exist_ok=True)
    
    test_main_cfg_path = os.path.join(test_config_dir, "main_test.yaml")
    test_model_cfg_path = os.path.join(test_config_dir, "model_test.yaml")
    test_risk_cfg_path = os.path.join(test_config_dir, "risk_test.yaml")

    # Initialize service (will create empty files if they don't exist)
    service = ConfigService(
        main_config_path=test_main_cfg_path,
        model_params_path=test_model_cfg_path,
        risk_limits_path=test_risk_cfg_path
    )

    # Test get_config (on initially empty or non-existent files)
    logger_main.info("\n--- Testing get_config on potentially empty files ---")
    try:
        main_c = service.get_config("main_config")
        logger_main.info(f"Initial main_config content: {main_c}")
    except Exception as e:
        logger_main.error(f"Error getting initial main_config: {e}")

    # Test update_config
    logger_main.info("\n--- Testing update_config ---")
    new_main_data = {"project_name": "TestProject API", "version": "v0.0.1", "paths": {"data_dir_raw": "./test_data"}}
    try:
        updated_main_c = service.update_config("main_config", new_main_data)
        logger_main.info(f"Updated main_config content: {updated_main_c}")
        
        # Verify by reading again
        reread_main_c = service.get_config("main_config")
        assert reread_main_c == new_main_data, "Reread config does not match updated data!"
        logger_main.info("Verified: Reread main_config matches updated data.")

    except Exception as e:
        logger_main.error(f"Error during update/verify main_config: {e}")

    # Test with an invalid config name
    logger_main.info("\n--- Testing with invalid config name ---")
    try:
        service.get_config("non_existent_config")
    except ValueError as e:
        logger_main.info(f"Caught expected ValueError for invalid config name: {e}")
    
    try:
        service.update_config("non_existent_config", {"key": "value"})
    except ValueError as e:
        logger_main.info(f"Caught expected ValueError for invalid config name (update): {e}")

    # Clean up dummy files and directory
    import shutil
    if os.path.exists(test_config_dir):
        shutil.rmtree(test_config_dir)
        logger_main.info(f"Cleaned up temporary test config directory: {test_config_dir}")

    logger_main.info("\nConfigService example usage complete.")

