from typing import List, Optional, Callable, Any
from .widget import Widget

class MenuItem(Widget):
    """A single item in a context menu."""

    def __init__(self, label: str, action: Optional[Callable[[], None]] = None, parent: Optional['ContextMenu'] = None):
        super().__init__(parent, name=f"MenuItem-{label}")
        self.label = label
        self.action = action
        self.submenu: Optional['ContextMenu'] = None
        self.checked: bool = False

    def set_submenu(self, menu: 'ContextMenu') -> None:
        """Attaches a submenu to this item."""
        self.submenu = menu
        menu.parent = self # Parent of submenu is the item? Or the menu?
                           # Usually submenu is a separate widget popping up.
                           # For structure, let's keep it as child.
        if menu not in self.children:
            self.children.append(menu)

    def trigger(self) -> None:
        """Executes the action if available."""
        if self.enabled and self.action:
            self.action()
        elif self.submenu:
            # In a real UI, this would show the submenu
            print(f"Opening submenu for {self.label}")

    def render(self) -> str:
        state = "[x]" if self.checked else "[ ]" # Simplification
        return f"{self.label}"

class ContextMenu(Widget):
    """A context menu containing a list of items."""

    def __init__(self, parent: Optional[Widget] = None, title: str = "Context Menu"):
        super().__init__(parent, name=f"ContextMenu-{title}")
        self.title = title
        self.items: List[MenuItem] = []

    def add_item(self, label: str, action: Optional[Callable[[], None]] = None) -> MenuItem:
        """Adds a new item to the menu."""
        item = MenuItem(label, action, parent=self)
        self.items.append(item)
        return item

    def add_separator(self) -> None:
        """Adds a separator (simulated as item with no action and specific label)."""
        self.add_item("---", None)

    def get_item(self, index: int) -> Optional[MenuItem]:
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def render(self) -> str:
        lines = [f"=== {self.title} ==="]
        for idx, item in enumerate(self.items):
            lines.append(f"{idx}. {item.render()}")
        return "\n".join(lines)

    def show(self) -> None:
        """Simulates showing the menu."""
        print(self.render())
