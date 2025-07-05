# Question 26: Name three types of objects in database and describe their roles.

**Status:** ✅ Completed | **Words:** 113 | **Generated:** 2025-07-04T00:10:00Z

---

## Answer

Relational databases revolve around a few core object types. First, tables are physical structures that persist rows of data organised in columns; they enforce data types, keys and constraints, forming the backbone of the schema. Second, views are virtual tables defined by SELECT queries; they expose subsets or joins of base tables, encapsulate business logic, and provide access-control granularity without duplicating data. Third, indexes are auxiliary lookup structures—typically B-trees or hash maps—that reorder selected columns so the engine can locate rows in logarithmic or constant time instead of scanning entire tables. Together these objects separate storage, presentation and performance concerns, allowing designers to tune a database for integrity, clarity and speed.

---
