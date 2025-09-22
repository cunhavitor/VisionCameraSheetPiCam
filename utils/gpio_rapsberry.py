try:
    import RPi.GPIO as GPIO  # type: ignore
    _GPIO_AVAILABLE = True
except Exception:
    GPIO = None  # type: ignore
    _GPIO_AVAILABLE = False


class RaspberryGPIO:
    """Lightweight GPIO reader wrapper.

    - Initializes the given pins as inputs with optional pull (UP/DOWN).
    - Provides `read_states()` -> {pin: bool} and `cleanup()`.
    - Safe to use off-Pi; methods degrade gracefully when GPIO is unavailable.
    """

    def __init__(self, pins, mode='BCM', pull='UP'):
        self.pins = list(pins or [])
        self.available = _GPIO_AVAILABLE and len(self.pins) > 0
        self._configured = False
        if not self.available:
            return
        try:
            if mode == 'BCM':
                GPIO.setmode(GPIO.BCM)
            else:
                GPIO.setmode(GPIO.BOARD)
            pud = GPIO.PUD_UP if str(pull).upper() == 'UP' else GPIO.PUD_DOWN
            for p in self.pins:
                try:
                    GPIO.setup(p, GPIO.IN, pull_up_down=pud)
                except Exception:
                    GPIO.setup(p, GPIO.IN)
            self._configured = True
        except Exception:
            self.available = False

    def read_states(self):
        """Return dict {pin: bool}. If unavailable, returns {}."""
        if not (self.available and self._configured):
            return {}
        states = {}
        for p in self.pins:
            try:
                states[p] = bool(GPIO.input(p))
            except Exception:
                states[p] = False
        return states

    def cleanup(self):
        if not (self.available and self._configured):
            return
        try:
            GPIO.cleanup(self.pins)
        except Exception:
            pass

