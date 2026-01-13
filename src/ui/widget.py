from typing import List, Optional, Any, Dict

class Widget:
    """Base class for all UI widgets."""

    def __init__(self, parent: Optional['Widget'] = None, name: str = "Widget"):
        self.parent = parent
        self.name = name
        self.children: List['Widget'] = []
        self.visible = True
        self.enabled = True
        self.position = (0, 0)
        self.size = (0, 0)
        self._styles: Dict[str, Any] = {}

        if parent:
            parent.add_child(self)

    def add_child(self, child: 'Widget') -> None:
        """Adds a child widget."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: 'Widget') -> None:
        """Removes a child widget."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def render(self) -> str:
        """
        Renders the widget.
        Returns a string representation for TUI/Debug purposes.
        """
        return f"[{self.name}]"

    def handle_event(self, event: Any) -> bool:
        """
        Handles an event.
        Returns True if the event was handled, False otherwise.
        """
        # Propagate to children first (bubbling or capturing strategy could be used)
        # Here we assume children might handle it first if it's targeted?
        # Or maybe this handles it.
        # For now, base implementation does nothing.
        return False

    def set_style(self, key: str, value: Any) -> None:
        self._styles[key] = value

    def get_style(self, key: str, default: Any = None) -> Any:
        return self._styles.get(key, default)

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' children={len(self.children)}>"
