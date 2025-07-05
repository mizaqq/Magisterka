# Question 12: Discuss the ISO/OSI RM model.

**Status:** ✅ Completed | **Words:** 149 | **Generated:** 2024-12-07T13:23:00Z

---

## Answer

The ISO/OSI Reference Model organises network functionality into seven hierarchical layers—Physical, Data Link, Network, Transport, Session, Presentation, and Application—providing a conceptual framework that standardises communication protocols and supports interoperability. Each layer offers specific services to the layer above while relying on the layer below, enabling modular design and independent evolution of technologies. The Physical Layer transmits raw bits over media; the Data Link Layer ensures frame integrity via MAC addressing and error detection; the Network Layer handles routing and logical addressing; the Transport Layer guarantees end-to-end reliability or sequencing (TCP) or offers best-effort delivery (UDP). Session management, data representation (encryption, compression), and user-facing services reside in layers five through seven, although many modern protocols collapse these distinctions. The OSI model clarifies vocabulary for debugging and protocol development, despite the Internet stack's pragmatic four-layer structure. It remains a pedagogical tool for understanding encapsulation, interface boundaries, and how complex networked systems interoperate.

---
