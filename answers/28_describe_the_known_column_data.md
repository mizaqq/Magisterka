# Question 28: Describe the known column data types.

**Status:** ✅ Completed | **Words:** 140 | **Generated:** 2025-07-04T00:12:00Z

---

## Answer

SQL column data types fall into several broad categories. Numeric types include INTEGER, BIGINT, DECIMAL and FLOAT for exact or approximate arithmetic. Character types VARCHAR, CHAR and TEXT store variable- or fixed-length strings, with collation rules governing sorting. Date and time types—DATE, TIME, TIMESTAMP and INTERVAL—record temporal values, optionally with timezone awareness. Boolean holds TRUE/FALSE/NULL, while binary types such as BYTEA or BLOB keep opaque byte streams like images. Modern systems add ENUM for controlled vocabularies, JSON/JSONB for semi-structured documents, spatial types (GEOMETRY, GEOGRAPHY) for GIS data and UUID for universally unique identifiers. Choosing the narrowest appropriate type lowers storage, improves index density and guards against invalid data.

---
