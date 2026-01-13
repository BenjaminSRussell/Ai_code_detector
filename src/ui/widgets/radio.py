from .base import Widget

class RadioGroup:
    """Manages a group of radio buttons."""

    def __init__(self):
        self.buttons = []
        self.selected_value = None

    def add_button(self, button):
        """Register a radio button with the group."""
        if button not in self.buttons:
            self.buttons.append(button)
            # If the button is checked, it becomes the selected one
            if button.checked:
                self.selected_value = button.value
                # Ensure other buttons are unchecked if this one claims to be checked
                # But wait, if multiple claim to be checked, the last one wins?
                # Let's enforce mutual exclusion based on arrival order if needed,
                # or just update the selected value.
                for btn in self.buttons:
                    if btn != button:
                        btn.checked = False

    def select(self, button):
        """Select a button in the group."""
        if button in self.buttons:
            self.selected_value = button.value
            for btn in self.buttons:
                btn.checked = (btn == button)

class RadioButton(Widget):
    """A radio button widget."""

    def __init__(self, label="", value=None, group=None, checked=False, **kwargs):
        """Initialize the RadioButton.

        Args:
            label (str): Text label.
            value (any): Value associated with this option.
            group (RadioGroup, optional): The group this button belongs to.
            checked (bool): Initial state.
            **kwargs: Arguments passed to Widget.
        """
        super().__init__(**kwargs)
        self.label = label
        self.value = value if value is not None else label
        self.group = group
        self.checked = checked

        if self.group:
            self.group.add_button(self)

    def select(self):
        """Select this radio button."""
        if self.group:
            self.group.select(self)
        else:
            self.checked = True

    def render(self):
        """Render the radio button.

        Returns:
            list[str]: Representation of the radio button.
        """
        icon = "(o)" if self.checked else "( )"
        return [f"{icon} {self.label}"]

    def handle_event(self, event):
        """Handle events. Selects on 'enter', 'space', or 'click'.

        Args:
            event: Event object.
        """
        if not hasattr(event, 'type'):
            return

        if event.type == 'key':
            if getattr(event, 'key', '') in ('enter', ' '):
                self.select()
        elif event.type == 'click':
            self.select()
