"""Tests for the Tooltip widget."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui.widgets import Widget, Tooltip

def test_tooltip_initialization():
    """Test initializing a tooltip."""
    tooltip = Tooltip(text="Hello")
    assert tooltip.text == "Hello"
    assert tooltip.visible is False
    assert tooltip.width == len("Hello") + 2
    assert tooltip.height == 1 + 2  # 1 line + 2 border

def test_tooltip_multiline():
    """Test multiline tooltip dimensions."""
    text = "Line 1\nLonger Line 2"
    tooltip = Tooltip(text=text)
    assert tooltip.width == len("Longer Line 2") + 2
    assert tooltip.height == 2 + 2

def test_tooltip_show_hide():
    """Test show and hide methods."""
    tooltip = Tooltip(text="Test")
    assert tooltip.visible is False

    tooltip.show()
    assert tooltip.visible is True

    tooltip.hide()
    assert tooltip.visible is False

def test_tooltip_render_hidden():
    """Test rendering when hidden."""
    tooltip = Tooltip(text="Test")
    tooltip.hide()
    assert tooltip.render() == []

def test_tooltip_render_visible():
    """Test rendering when visible."""
    tooltip = Tooltip(text="Test")
    tooltip.show()
    rendered = tooltip.render()

    assert len(rendered) == 3 # top, content, bottom
    assert rendered[0] == "┌────┐"
    assert rendered[1] == "│Test│"
    assert rendered[2] == "└────┘"

def test_tooltip_positioning():
    """Test automatic positioning relative to target."""
    parent = Widget(width=100, height=100)
    target = Widget(x=10, y=10, width=20, height=5, parent=parent)

    # Tooltip sharing the same parent
    tooltip = Tooltip(text="Info", target=target, parent=parent)

    tooltip.show()

    # Expect tooltip to be positioned below target
    # x = target.x + 2 = 12
    # y = target.y + target.height = 10 + 5 = 15
    assert tooltip.x == 12
    assert tooltip.y == 15

if __name__ == "__main__":
    # Run tests manually if executed directly
    test_tooltip_initialization()
    test_tooltip_multiline()
    test_tooltip_show_hide()
    test_tooltip_render_hidden()
    test_tooltip_render_visible()
    test_tooltip_positioning()
    print("All tooltip tests passed!")
