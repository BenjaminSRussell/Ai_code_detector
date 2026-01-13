"""Tests for UI Context Menu System."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui.widget import Widget
from ui.context_menu import ContextMenu, MenuItem

def test_widget_hierarchy():
    """Test basic widget parent-child relationship."""
    parent = Widget(name="Parent")
    child = Widget(parent=parent, name="Child")

    assert child in parent.children
    assert child.parent == parent

    parent.remove_child(child)
    assert child not in parent.children
    assert child.parent is None

def test_context_menu_creation():
    """Test creating a context menu and adding items."""
    menu = ContextMenu(title="Edit")
    item1 = menu.add_item("Cut")
    item2 = menu.add_item("Copy")

    assert len(menu.items) == 2
    assert item1.label == "Cut"
    assert item2.label == "Copy"

    # Check hierarchy
    assert item1 in menu.children
    assert item1.parent == menu

def test_context_menu_action():
    """Test triggering an action on a menu item."""
    triggered = False
    def on_click():
        nonlocal triggered
        triggered = True

    menu = ContextMenu()
    item = menu.add_item("Click Me", action=on_click)

    item.trigger()
    assert triggered

def test_submenu():
    """Test attaching a submenu."""
    root = ContextMenu(title="Root")
    sub = ContextMenu(title="Sub")

    item = root.add_item("Nested")
    item.set_submenu(sub)

    assert item.submenu == sub
    assert sub.parent == item

def test_rendering():
    """Test simulated rendering string."""
    menu = ContextMenu(title="Main")
    menu.add_item("One")
    menu.add_item("Two")

    render_output = menu.render()
    assert "=== Main ===" in render_output
    assert "0. One" in render_output
    assert "1. Two" in render_output

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
