"""Tooltip widget implementation."""

from typing import Optional, List
from .base import Widget

class Tooltip(Widget):
    """A tooltip widget that displays text when triggered."""

    def __init__(self, text: str, target: Optional[Widget] = None, **kwargs):
        """Initialize the tooltip.

        Args:
            text: The text to display in the tooltip.
            target: The widget this tooltip is attached to.
            **kwargs: Additional arguments for the Widget constructor.
        """
        super().__init__(**kwargs)
        self.text = text
        self.target = target
        self.visible = False  # Tooltips are hidden by default

        # Calculate dimensions based on text
        lines = self.text.split('\n')
        self.width = max(len(line) for line in lines) + 2  # +2 for padding/border
        self.height = len(lines) + 2  # +2 for padding/border

    def set_target(self, target: Widget):
        """Set the target widget for this tooltip."""
        self.target = target

    def show(self):
        """Show the tooltip.

        Calculates position relative to target if set.
        """
        if self.target:
            # Position tooltip below the target by default
            # Use absolute positions to calculate where to place it in the root container usually,
            # but here we might just set relative x/y if we share the same parent or are a child of root.
            # For simplicity, let's assume we want to place it relative to the target's position.
            # But if the tooltip is a child of the screen/root, we need absolute coordinates.

            # If the tooltip is added to the same parent as the target:
            if self.parent and self.parent == self.target.parent:
                self.x = self.target.x + 2 # slightly offset
                self.y = self.target.y + self.target.height

            # If the tooltip is a child of the target (which is common for encapsulation but bad for z-index),
            # then x=0, y=height would mean relative to target.
            # But usually tooltips are top-level.
            # Let's assume manual positioning logic for now or simple "below target".
            pass

        self.visible = True

    def hide(self):
        """Hide the tooltip."""
        self.visible = False

    def render(self) -> List[str]:
        """Render the tooltip as a list of strings (TUI style)."""
        if not self.visible:
            return []

        lines = self.text.split('\n')
        width = self.width

        # Create a simple box with ASCII characters
        result = []
        result.append('┌' + '─' * (width - 2) + '┐')
        for line in lines:
            padded_line = line.ljust(width - 2)
            result.append(f'│{padded_line}│')
        result.append('└' + '─' * (width - 2) + '┘')

        return result
