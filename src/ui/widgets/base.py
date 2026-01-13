class Widget:
    """Base class for all UI widgets."""

    def __init__(self, parent=None, x=0, y=0, width=10, height=1):
        """Initialize the widget.

        Args:
            parent (Widget, optional): The parent widget.
            x (int): X coordinate relative to parent.
            y (int): Y coordinate relative to parent.
            width (int): Width of the widget.
            height (int): Height of the widget.
        """
        self.parent = parent
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.children = []
        self._focused = False

        if parent:
            parent.add_child(self)

    def add_child(self, child):
        """Add a child widget."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child):
        """Remove a child widget."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def render(self):
        """Render the widget.

        Returns:
            list[str]: A list of strings representing the rows of the widget.
        """
        raise NotImplementedError("Subclasses must implement render()")

    def handle_event(self, event):
        """Handle an input event.

        Args:
            event: An object or dictionary representing the event.
                   Expected to have 'type' and optionally other attributes.
        """
        pass

    @property
    def focused(self):
        return self._focused

    @focused.setter
    def focused(self, value):
        self._focused = value

    def focus(self):
        """Set focus to this widget."""
        self.focused = True

    def blur(self):
        """Remove focus from this widget."""
        self.focused = False
