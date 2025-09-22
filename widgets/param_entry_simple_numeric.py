import customtkinter as ctk

def create_param_entry(parent, label_text, var, command=None, step=1, min_value=None, max_value=None):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(pady=2, fill="x")

    label = ctk.CTkLabel(frame, text=label_text)
    label.pack(side="left", padx=5)

    # Separador visual como linha expandida
    line = ctk.CTkLabel(frame, text="─" * 10)  # ou use "—" ou "-"
    line.pack(side="left", expand=True, fill="x", padx=5)

    percent_label = ctk.CTkLabel(frame, text="%")
    percent_label.pack(side="right")

    entry = ctk.CTkEntry(frame, textvariable=var, width=60)
    entry.pack(side="right", padx=5)

    def callback(*args):
        print(f"callback called: {var.get()}")
        if command and not getattr(parent.winfo_toplevel(), "silent_mode", False):
            command()

    var.trace_add("write", callback)

    def on_mouse_wheel(event):
        try:
            current_val = int(var.get())
        except Exception:
            current_val = 0

        if event.delta > 0:
            new_val = current_val + step
        else:
            new_val = current_val - step

        if min_value is not None:
            new_val = max(new_val, min_value)
        if max_value is not None:
            new_val = min(new_val, max_value)

        var.set(str(new_val))  # <- garante que continua sendo string

    entry.bind("<MouseWheel>", on_mouse_wheel)
    entry.bind("<Button-4>", lambda e: on_mouse_wheel(type('Event', (object,), {'delta': 1})()))
    entry.bind("<Button-5>", lambda e: on_mouse_wheel(type('Event', (object,), {'delta': -1})()))

    return frame
