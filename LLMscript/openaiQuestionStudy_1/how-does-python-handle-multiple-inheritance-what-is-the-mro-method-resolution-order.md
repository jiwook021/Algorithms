# How does Python handle multiple inheritance? What is the MRO (Method Resolution Order)?

Python supports multiple inheritance, where a class can inherit features from more than one parent class. This can lead to complex inheritance structures and potential conflicts if the same method or attribute is defined in multiple parent classes. Python addresses these complexities through a methodology known as the Method Resolution Order (MRO).

### Multiple Inheritance

In Python, a class can inherit from multiple parent classes:

```python
class Base1:
    def method(self):
        print("Method in Base1")

class Base2:
    def method(self):
        print("Method in Base2")

class Child(Base1, Base2):
    pass

obj = Child()
obj.method()
```

In this example, `Child` inherits from both `Base1` and `Base2`. Both base classes define a method called `method`. The output of this code will depend on the MRO of `Child`.

### Method Resolution Order (MRO)

The MRO determines the order in which Python looks for methods in a hierarchy of classes. With multiple inheritance, the MRO ensures that every class is visited only once and that if a class inherits from multiple classes, the order respects the order specified in the declaration.

Python computes the MRO using the C3 linearization (or C3 superclass linearization) algorithm. This algorithm provides a way to linearly order the classes for lookup purposes.

#### How C3 Linearization Works:

1. **Construct a list for each class**: A list beginning with the class itself followed by its base classes as specified in its declaration, and recursively including the base classes of those bases.

2. **Merge the lists**: The lists are merged according to these rules:
   - Choose the first head of the lists that does not appear in the tail (i.e., anywhere except the first position) of any lists.
   - Remove this element from all lists and append it to the result.
   - Repeat until all elements are removed.
   - If a valid linearization is not possible, Python will raise an error.

#### Example of MRO Calculation:

For the `Child` class above, assuming the classes `Base1` and `Base2` do not inherit from other classes, the MRO lists would be:

- `Child`: [Child] + merge(MRO(Base1), MRO(Base2), [Base1, Base2])
- `Base1`: [Base1, object]
- `Base2`: [Base2, object]

Merging these:

- Start with `[Child]`.
- Merge `[Base1, object]`, `[Base2, object]`, `[Base1, Base2]`.
- Select `Base1` (it does not appear in the tail of any list).
- Next, select `Base2`.
- Finally, add `object`.

The MRO is: `[Child, Base1, Base2, object]`.

This means Python will first look for methods in `Child`, then `Base1`, then `Base2`, and finally in `object` (the base class for all classes).

You can view the MRO of any class in Python using the `__mro__` attribute or the `mro()` method:

```python
print(Child.__mro__)
# or
print(Child.mro())
```

Understanding MRO is crucial for designing classes in Python that utilize multiple inheritance, as it defines how methods and attributes are inherited and resolved, thereby preventing potential conflicts or unexpected behaviors.