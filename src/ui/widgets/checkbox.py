from .base import Widget

class CheckBox(Widget):
    """A checkbox widget."""

    def __init__(self, label="", checked=False, **kwargs):
        """Initialize the CheckBox.

        Args:
            label (str): Text label for the checkbox.
            checked (bool): Initial state.
            **kwargs: Arguments passed to Widget.
        """
        super().__init__(**kwargs)
        self.label = label
        self.checked = checked

    def toggle(self):
        """Toggle the checked state."""
        self.checked = not self.checked

    def render(self):
        """Render the checkbox.

        Returns:
            list[str]: Representation of the checkbox.
        """
        icon = "[x]" if self.checked else "[ ]"
        return [f"{icon} {self.label}"]

    def handle_event(self, event):
        """Handle events. Toggles on 'enter' or 'space' key press or 'click'.

        Args:
            event: Event object.
        """
        if not hasattr(event, 'type'):
            return

        if event.type == 'key':
            if getattr(event, 'key', '') in ('enter', ' '):
                self.toggle()
        elif event.type == 'click':
            self.toggle()
