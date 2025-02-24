import re
import ast
from ._utils import strip_markers

_NO_VALUE = object()

class VariableStack:
    """This represents the variables scope stack of a Guidance program."""

    def __init__(self, stack, executor):
        """Build a new variable stack object with the given stack and program executor."""
        self._stack = stack
        self._executor = executor

    def push(self, variables):
        self._stack.append(variables)

    def pop(self):
        out = self._stack.pop()

        # if we are popping a _prefix variable state we need to update the display
        if "_prefix" in self._stack[-1]:
            self._executor.program.update_display()
        
        return out

    def __getitem__(self, key):
        return self.get(key)
    
    def get(self, name, default_value=None):

        # prefix is a special variable that returns the current prefix without the marker tags
        if name == "prefix":
            return strip_markers(self.get("_prefix", ""))

        parts = re.split(r"\.|\[", name)
        for variables in reversed(self._stack):
            curr_pos = variables
            found = True
            for part in parts:
                if part.endswith("]"):
                    var_part = ast.literal_eval(part[:-1])
                else:
                    var_part = part
                try:
                    next_pos = curr_pos[var_part]
                    next_found = True
                except KeyError:
                    next_found = False
                if next_found:
                    curr_pos = next_pos
                else:
                    found = False
                    break
            if found:
                return curr_pos
        return default_value # variable not found

    def __contains__(self, name):
        return self.get(name, _NO_VALUE) != _NO_VALUE

    def __setitem__(self, key, value):
        parts = re.split(r"\.|\[", key)
        found = True
        changed = True
        for variables in reversed(self._stack):
            curr_pos = variables
            found = True
            for part in parts:
                if part.endswith("]"):
                    var_part = ast.literal_eval(part[:-1])
                else:
                    var_part = part
                try:
                    next_pos = curr_pos[var_part]
                    next_found = True
                except KeyError:
                    next_found = False
                
                if next_found:
                    if part == parts[-1]:
                        changed = curr_pos[var_part] != value
                        curr_pos[var_part] = value
                        break
                    else:
                        curr_pos = next_pos
                else:
                    if part == parts[-1] and len(parts) > 1: # setting a new property
                        curr_pos[var_part] = value
                    else:
                        found = False
                    break
            if found:
                break
        if not found:
            assert len(parts) == 1, "Can't set a property of a non-existing variable: " + key
            self._stack[0][key] = value
        
        # if we changed the _prefix variable, update the display
        if changed and key == "_prefix" and not self.get("_no_display", False):
            self._executor.program.update_display()

    def copy(self):
        return VariableStack(self._stack.copy(), self._executor)