Imagine you're building with LEGOs.

**Without Dependency Injection (DI):**

1. **Hardcoded connections:** You build a big spaceship.  Inside, the engine is directly glued to the cockpit.  The power generator is directly glued to the engine.  Everything is stuck together.

2. **Difficult changes:**  If you want to upgrade the engine, you have to rip apart the whole spaceship, unglue the old engine, and glue in the new one.  You risk breaking other parts in the process.  Even a small change is a huge hassle.

**With Dependency Injection (DI):**

1. **Separate parts:** You build the spaceship with separate modules: a cockpit module, an engine module, and a power generator module.  Each module has connection points (like LEGO studs).

2. **Plugging in modules:** You connect the modules using these connection points.  You can easily swap out the engine module for a more powerful one, without touching the cockpit or power generator.  If a module is broken, you replace only that module.

**How this relates to code:**

In programming, "modules" are classes or functions.  "Connections" are how they interact.

* **Without DI:** Classes directly create the objects they need (hardcoded connections).  This makes the code tightly coupled – changing one part requires changing many others.

* **With DI:** Classes receive the objects they need as input (like plugging in LEGO modules). This is Dependency Injection.  It "injects" the dependencies into the class instead of the class creating them itself.

**Steps to understand DI's improvement in maintainability:**

1. **Loose Coupling:** DI separates concerns.  Classes are independent, making changes easier. You can modify one class without affecting others, as long as the interfaces remain the same.

2. **Reusability:** A class becomes reusable in different contexts because its dependencies are provided externally.  The same engine module can be used in different spaceships.

3. **Testability:**  It's easier to test individual classes because you can provide mock dependencies (fake LEGO engines for testing).  You don't need to set up the whole spaceship to test the cockpit.

4. **Maintainability:**  Overall, the code is easier to maintain, debug, and extend because of loose coupling, reusability, and testability.  Changes are localized, reducing the risk of introducing bugs elsewhere.


In short: DI makes your code more flexible and robust, like a spaceship built with easily interchangeable LEGO modules instead of glued-together parts.
