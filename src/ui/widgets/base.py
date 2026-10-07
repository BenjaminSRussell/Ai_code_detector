"""Base Widget class for the UI system."""

from typing import List, Optional, Any

class Widget:
    """Base class for all UI widgets."""

    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0, parent: Optional['Widget'] = None):
        """Initialize the widget.

        Args:
            x: X coordinate relative to parent.
            y: Y coordinate relative to parent.
            width: Width of the widget.
            height: Height of the widget.
            parent: Parent widget.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.parent = parent
        self.children: List['Widget'] = []
        self.visible = True
        self.focused = False

        if parent:
            parent.add_child(self)

    def add_child(self, child: 'Widget'):
        """Add a child widget."""
        if child not in self.children:
            if child.parent:
                child.parent.remove_child(child)
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: 'Widget'):
        """Remove a child widget."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def render(self) -> Any:
        """Render the widget. To be implemented by subclasses."""
        pass

    def handle_event(self, event: Any) -> bool:
        """Handle an event. To be implemented by subclasses.

        Returns:
            True if the event was handled, False otherwise.
        """
        return False

    def get_absolute_position(self) -> tuple[int, int]:
        """Get the absolute (screen) coordinates of the widget."""
        if self.parent:
            px, py = self.parent.get_absolute_position()
            return px + self.x, py + self.y
        return self.x, self.y
