import os
from ruamel.yaml import YAML, CommentedMap
from febid.start import Starter
from febid.logging_config import setup_logger
from febid.simulation_context import SimulationParameters

logger = setup_logger(__name__)

class SessionManager:
    """
    Handles loading, saving, and updating session (interface configuration) parameters.
    Hybrid: stores config as CommentedMap for comment preservation, but can convert to/from SimulationParameters for validation and type safety.
    """
    def __init__(self, default_config_stub=None):
        self.params = CommentedMap()
        self._params_dataclass = None
        self.starter = Starter()
        if default_config_stub is None:
            self.default_config_stub = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_session_stub.yml')
        else:
            self.default_config_stub = default_config_stub

    def load(self, filename):
        try:
            with open(filename, mode='rb') as f:
                yml = YAML()
                params = yml.load(f)
                self.params = params if params is not None else CommentedMap()
        except FileNotFoundError:
            logger.exception('Session file not found')
            raise

    def create(self, params):
        self.load_empty()
        self.set_all_params(params)

    def load_empty(self):
        self.load(self.default_config_stub)

    def save(self, filename):
        yml = YAML()
        with open(filename, mode='wb') as f:
            yml.dump(self.params, f)

    def set_param(self, name, value):
        self.params[name] = value

    def set_all_params(self, params):
        """
        Set all parameters from the current params dict.
        """
        for name, value in params.items():
            self.set_param(name, value)

    def get_param(self, name, default=None):
        return self.params.get(name, default)

    def to_dataclass(self) -> SimulationParameters:
        """
        Convert current params to a SimulationParameters dataclass (for validation/type safety).
        Ignores unknown fields.
        """
        # Only pass fields that exist in the dataclass
        dc_fields = {k: self.params[k] for k in SimulationParameters.__dataclass_fields__ if k in self.params}
        self._params_dataclass = SimulationParameters(**dc_fields)
        return self._params_dataclass

    def validate(self):
        """
        Validate current parameters using the dataclass. Raises if invalid.
        """
        params_dc = self.to_dataclass()
        params_dc.validate()

    def start(self, module='febid', **kwargs):
        # Optionally validate before starting
        self.validate()
        if module == 'febid':
            # Pass dataclass to starter if needed, else use dict
            self.starter.params = self.to_dataclass()
            return self.starter.start()
        elif module == 'monte_carlo':
            self.starter.params = self.to_dataclass()
            self.starter.start_mc(**kwargs)
        else:
            raise ValueError(f'Unknown module: {module}')

    def stop(self):
        self.starter.stop()

