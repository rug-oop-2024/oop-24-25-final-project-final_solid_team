#!/bin/env python

import sys
import pyperclip

print(pyperclip.paste())

index = 1
getters_and_setter = ""
while True:
    try:
        attr = sys.argv[index]
        index += 1

        function = (
            f"@property\n"
            f"def {attr}(self) -> _type_:\n"
            f"    return self._{attr}\n\n"
        )
        getters_and_setter += function
        print(f"Added {function} to clipboard")
    except:
        print(f"You passed {index - 1} attributes")
        break


pyperclip.copy(getters_and_setter)
