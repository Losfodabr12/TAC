import contextlib
import threading


class LearningDisabled(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class LearningGate:
    """Mesmo lock protege o desligamento, commits de conhecimento e optimizer.step."""
    def __init__(self, config, db):
        self.config, self.db = config, db
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.paused = threading.Event()

    @property
    def enabled(self):
        return self.config.get("learning_enabled")

    def set_enabled(self, value):
        with self.lock:
            self.config.update(learning_enabled=bool(value))
            if not value:
                self.cancel.set()
                self.paused.clear()
            self.db.event("aprendizado", "ativado" if value else "desativado")

    @contextlib.contextmanager
    def writing(self):
        with self.lock:
            if not self.enabled:
                raise LearningDisabled("Aprendizado desativado. O material permanece na fila.")
            yield

    def begin(self):
        with self.lock:
            if not self.enabled:
                raise LearningDisabled("Ative o aprendizado antes de iniciar esta operação.")
            self.cancel.clear()
            self.paused.clear()

    def check(self):
        if self.cancel.is_set() or not self.enabled:
            raise Cancelled("Interrupção solicitada; progresso confirmado foi preservado.")

    def wait_if_paused(self):
        while self.paused.is_set():
            self.check()
            self.cancel.wait(0.1)
        self.check()
