import sys
import os


# Show a dark themed launcher window — user can drag a folder onto it or click Browse.
def pick_folder(prompt: str = "Select the patient DICOM folder") -> str:
    # CLI argument / drag-and-drop directly onto terminal — use immediately
    if len(sys.argv) > 1:
        path = sys.argv[1].strip().strip("'\"")  # strip shell quoting
        path = os.path.normpath(path)
        if os.path.isdir(path):
            return path

    return _launcher_window(prompt)


def _launcher_window(prompt: str) -> str:
    from tkinter import filedialog

    # colour palette matching the visualizer
    BG      = "#0f0f0f"
    ZONE_BG = "#1a1a1a"
    ZONE_BD = "#333333"
    ACCENT  = "#e05252"
    FG      = "#ffffff"
    FG_DIM  = "#888888"

    result = {"path": None}

    # TkinterDnD.Tk() is required for OS-level drag-and-drop to work
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        dnd_ok = True
    except ImportError:
        import tkinter as tk
        root = tk.Tk()
        dnd_ok = False

    import tkinter as tk  # still needed for widgets

    root.title("ProjectAnti — CAC Scorer")
    root.configure(bg=BG)
    root.resizable(False, False)

    # centre on screen
    w, h = 520, 340
    root.geometry(f"{w}x{h}+{(root.winfo_screenwidth()  - w) // 2}"
                            f"+{(root.winfo_screenheight() - h) // 2}")

    # title
    tk.Label(root, text="CAC Scorer", font=("Helvetica", 22, "bold"),
             bg=BG, fg=FG).pack(pady=(36, 4))
    tk.Label(root, text="Coronary Artery Calcium — Agatston Pipeline",
             font=("Helvetica", 11), bg=BG, fg=FG_DIM).pack()

    # drop zone frame
    zone = tk.Frame(root, bg=ZONE_BG, bd=0, highlightthickness=2,
                    highlightbackground=ZONE_BD, width=420, height=110)
    zone.pack(pady=24)
    zone.pack_propagate(False)

    tk.Label(zone, text="📂", font=("Helvetica", 28),
             bg=ZONE_BG, fg=FG_DIM).place(relx=0.5, rely=0.35, anchor="center")

    hint = "Drag a patient DICOM folder here" if dnd_ok else "Click Browse to select a folder"
    zone_label = tk.Label(zone, text=hint, font=("Helvetica", 12),
                          bg=ZONE_BG, fg=FG_DIM)
    zone_label.place(relx=0.5, rely=0.72, anchor="center")

    def _activate_zone(colour):
        zone.config(highlightbackground=colour)
        zone_label.config(fg=colour)

    # wire up drag-and-drop on the whole window (TkinterDnD.Tk handles it globally)
    if dnd_ok:
        def _on_drop(event):
            path = event.data.strip().strip("{}")  # tkdnd wraps paths in braces
            path = os.path.normpath(path)
            if os.path.isdir(path):
                result["path"] = path
                root.destroy()
            else:
                _activate_zone("#e05252")
                zone_label.config(text="That's not a folder — try again")

        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", _on_drop)
        root.dnd_bind("<<DragEnter>>", lambda e: _activate_zone(ACCENT))
        root.dnd_bind("<<DragLeave>>", lambda e: _activate_zone(ZONE_BD))

    # browse button
    def _browse():
        path = filedialog.askdirectory(title=prompt, parent=root)
        if path:
            result["path"] = os.path.normpath(path)
            root.destroy()

    btn = tk.Frame(root, bg=ACCENT, cursor="hand2")
    btn.pack()
    lbl = tk.Label(btn, text="  Browse Folder  ", font=("Helvetica", 13, "bold"),
                   bg=ACCENT, fg=FG, padx=16, pady=8)
    lbl.pack()
    btn.bind("<Button-1>", lambda e: _browse())
    lbl.bind("<Button-1>",  lambda e: _browse())

    # hover effect on button
    for widget in (btn, lbl):
        widget.bind("<Enter>", lambda e: [btn.config(bg="#c03e3e"), lbl.config(bg="#c03e3e")])
        widget.bind("<Leave>", lambda e: [btn.config(bg=ACCENT),    lbl.config(bg=ACCENT)])

    root.mainloop()

    if not result["path"]:
        print("No folder selected. Exiting.")
        sys.exit(1)

    return result["path"]
