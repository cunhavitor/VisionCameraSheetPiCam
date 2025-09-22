# defect_param_properties.py

class DefectParamPropertiesMixin:

    # DARK THRESHOLD
    def __init__(self):
        self.min_defect_area_var = None
        self.dark_gradient_threshold_var = None
        self.bright_iterations_var = None
        self.bright_kernel_var = None
        self.dark_iterations_var = None
        self.dark_kernel_var = None
        self.red_threshold_var = None
        self.blue_threshold_var = None
        self.bright_threshold_var = None
        self.dark_threshold_var = None

    def _safe_update_preview(self):
        if hasattr(self, "_update_preview"):
            self._update_preview()

    @property
    def dark_threshold(self):
        return int(self.dark_threshold_var.get())

    @dark_threshold.setter
    def dark_threshold(self, value):
        self.dark_threshold_var.set(str(value))
        self._safe_update_preview()

    @property
    def bright_threshold(self):
        return int(self.bright_threshold_var.get())

    @bright_threshold.setter
    def bright_threshold(self, value):
        self.bright_threshold_var.set(str(value))
        self._safe_update_preview()

    @property
    def blue_threshold(self):
        return int(self.blue_threshold_var.get())

    @blue_threshold.setter
    def blue_threshold(self, value):
        self.blue_threshold_var.set(str(value))
        self._safe_update_preview()

    @property
    def red_threshold(self):
        return int(self.red_threshold_var.get())

    @red_threshold.setter
    def red_threshold(self, value):
        self.red_threshold_var.set(str(value))
        self._safe_update_preview()

    @property
    def dark_morph_kernel_size(self):
        return int(self.dark_kernel_var.get())

    @dark_morph_kernel_size.setter
    def dark_morph_kernel_size(self, value):
        self.dark_kernel_var.set(str(value))
        self._safe_update_preview()

    @property
    def dark_morph_iterations(self):
        return int(self.dark_iterations_var.get())

    @dark_morph_iterations.setter
    def dark_morph_iterations(self, value):
        self.dark_iterations_var.set(str(value))
        self._safe_update_preview()

    @property
    def bright_morph_kernel_size(self):
        return int(self.bright_kernel_var.get())

    @bright_morph_kernel_size.setter
    def bright_morph_kernel_size(self, value):
        self.bright_kernel_var.set(str(value))
        self._safe_update_preview()

    @property
    def bright_morph_iterations(self):
        return int(self.bright_iterations_var.get())

    @bright_morph_iterations.setter
    def bright_morph_iterations(self, value):
        self.bright_iterations_var.set(str(value))
        self._safe_update_preview()

    @property
    def dark_gradient_threshold(self):
        return int(self.dark_gradient_threshold_var.get())

    @dark_gradient_threshold.setter
    def dark_gradient_threshold(self, value):
        self.dark_gradient_threshold_var.set(str(value))
        self._safe_update_preview()

    @property
    def min_defect_area(self):
        return int(self.min_defect_area_var.get())

    @min_defect_area.setter
    def min_defect_area(self, value):
        self.min_defect_area_var.set(str(value))
        self._safe_update_preview()
