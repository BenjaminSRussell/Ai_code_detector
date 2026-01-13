import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ui.widgets import Widget, CheckBox, RadioButton, RadioGroup

class MockEvent:
    def __init__(self, type, key=None):
        self.type = type
        self.key = key

def test_widget_base():
    """Test generic widget functionality."""
    parent = Widget(width=100, height=100)
    child = Widget(parent=parent, x=10, y=10)

    assert child in parent.children
    assert child.parent == parent
    assert child.x == 10

    parent.remove_child(child)
    assert child not in parent.children
    assert child.parent is None

    with pytest.raises(NotImplementedError):
        child.render()

def test_checkbox():
    """Test CheckBox widget."""
    cb = CheckBox("Test Checkbox")
    assert not cb.checked
    assert "[ ] Test Checkbox" in cb.render()[0]

    cb.toggle()
    assert cb.checked
    assert "[x] Test Checkbox" in cb.render()[0]

    # Test event handling
    event = MockEvent("key", "enter")
    cb.handle_event(event)
    assert not cb.checked  # Toggled back to False

def test_radio_group():
    """Test RadioGroup and RadioButton."""
    group = RadioGroup()

    rb1 = RadioButton("Option 1", value=1, group=group)
    rb2 = RadioButton("Option 2", value=2, group=group)
    rb3 = RadioButton("Option 3", value=3, group=group, checked=True)

    # Initial state: rb3 is checked
    assert not rb1.checked
    assert not rb2.checked
    assert rb3.checked
    assert group.selected_value == 3

    # Select rb1 via method
    rb1.select()
    assert rb1.checked
    assert not rb2.checked
    assert not rb3.checked
    assert group.selected_value == 1

    # Select rb2 via event
    event = MockEvent("click")
    rb2.handle_event(event)
    assert not rb1.checked
    assert rb2.checked
    assert not rb3.checked
    assert group.selected_value == 2

def test_radio_no_group():
    """Test RadioButton without a group."""
    rb = RadioButton("Independent")
    assert not rb.checked

    rb.select()
    assert rb.checked
