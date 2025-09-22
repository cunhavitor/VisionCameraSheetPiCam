# widgets/param_entry.py

import customtkinter as ctk

def _validate_numeric(value):
    # Permite números inteiros e vazios (para apagar temporariamente)
    return value == "" or value.isdigit()

def create_param_entry(parent_frame, label_text, var_obj, bind_command=None, master_widget=None):
    """
    Cria um par Label + Entry num frame interno com validação numérica.

    Args:
        parent_frame: widget pai onde o frame será colocado.
        label_text (str): texto a exibir no label.
        var_obj (tk.StringVar ou ctk.StringVar): variável ligada ao Entry.
        bind_command (função): função a executar ao sair do campo (FocusOut).
        master_widget: widget que possui o método register (normalmente `self`).

    Returns:
        (label, entry): widgets criados.
    """
    frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
    frame.pack(fill="x", pady=5)

    label = ctk.CTkLabel(frame, text=label_text)
    label.pack(side="left", padx=(0, 10))

    entry = ctk.CTkEntry(frame,
                         textvariable=var_obj,
                         fg_color="white",
                         text_color="black",
                         width=50)
    entry.pack(side="right")

    if master_widget:
        entry.configure(validate="key", validatecommand=(master_widget.register(_validate_numeric), '%P'))

    entry.bind("<FocusOut>", bind_command)

    return label, entry
