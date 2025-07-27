# src/agents/base_agent.py
import logging

class BaseAgent:
    """
    Optional base class for common agent functionalities.
    This can include things like:
    - Common logging setup.
    - Configuration loading.
    - Interface definitions (though Python's duck typing often suffices).
    """
    def __init__(self, agent_name: str, config: dict = None):
        """
        Initializes the BaseAgent.

        Args:
            agent_name (str): The name of the agent, used for logging.
            config (dict, optional): Configuration dictionary for the agent. Defaults to None.
        """
        self.agent_name = agent_name
        self.config = config if config is not None else {}
        self.logger = logging.getLogger(f"RLTradingPlatform.Agent.{self.agent_name}")
        self.logger.info(f"{self.agent_name} initialized.")

    def load_config(self, config_path: str):
        """
        Placeholder for loading agent-specific configuration.
        Actual implementation might use a utility function.

        Args:
            config_path (str): Path to the configuration file.
        """
        # TODO: Implement actual config loading logic, possibly using a shared utility.
        self.logger.info(f"Loading config from {config_path} (not yet implemented).")
        pass

    def run(self, *args, **kwargs):
        """
        A common method that derived agents might implement to perform their primary task.
        This forces a common entry point, but specific arguments will vary.
        """
        raise NotImplementedError(f"`run` method not implemented in {self.agent_name}")

    def get_status(self) -> dict:
        """
        Returns the current status of the agent.
        Useful for monitoring and orchestration.
        """
        return {"agent_name": self.agent_name, "status": "idle", "config": self.config}

if __name__ == '__main__':
    # Example usage (mainly for testing the base class itself)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    base_agent = BaseAgent(agent_name="TestBaseAgent", config={"param1": "value1"})
    print(base_agent.get_status())
    
    try:
        base_agent.run()
    except NotImplementedError as e:
        print(f"Caught expected error: {e}")
