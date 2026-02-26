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
        """Initialize session storage, defaults, and starter interface.

        :param default_config_stub: Path to default session template file.
        :type default_config_stub: str
        :return: None
        """
        self.params = CommentedMap()
        self._params_dataclass = None
        self.starter = Starter()
        if default_config_stub is None:
            self.default_config_stub = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_session_stub.yml')
        else:
            self.default_config_stub = default_config_stub

    def load(self, filename):
        """Load session parameters from YAML file.

        :param filename: Session file path.
        :type filename: str
        :return: None
        """
        try:
            with open(filename, mode='rb') as f:
                yml = YAML()
                params = yml.load(f)
                self.params = params if params is not None else CommentedMap()
        except FileNotFoundError:
            logger.exception('Session file not found')
            raise

    def create(self, params):
        """Create session from default template and override with provided parameters.

        :param params: Parameter mapping to apply.
        :type params: dict
        :return: None
        """
        self.load_empty()
        self.set_all_params(params)

    def load_empty(self):
        """Load default session stub.

        :return: None
        """
        self.load(self.default_config_stub)

    def save(self, filename):
        """Save current session parameters to YAML file.

        :param filename: Output session file path.
        :type filename: str
        :return: None
        """
        yml = YAML()
        with open(filename, mode='wb') as f:
            yml.dump(self.params, f)

    def set_param(self, name, value):
        """Set a single session parameter.

        :param name: Parameter name.
        :type name: str
        :param value: Parameter value.
        :type value: object
        :return: None
        """
        self.params[name] = value

    def set_all_params(self, params):
        """
        Set all parameters from the current params dict.
        """
        for name, value in params.items():
            self.set_param(name, value)

    def get_param(self, name, default=None):
        """Return one parameter value with an optional fallback.

        :param name: Parameter name.
        :type name: str
        :param default: Fallback value when parameter is not present.
        :type default: object
        :return: Stored value or default.
        """
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
        """Validate configuration and start the selected simulation module.

        :param module: Module selector (`febid` or `monte_carlo`).
        :type module: str
        :param kwargs: Extra module-specific runtime arguments.
        :type kwargs: dict
        :return: Simulation manager for FEBID mode, otherwise None.
        """
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
        """Stop the currently running simulation via starter interface.

        :return: None
        """
        self.starter.stop()

