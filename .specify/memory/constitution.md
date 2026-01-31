# kcuda Constitution

<!--
═══════════════════════════════════════════════════════════════════════════
SYNC IMPACT REPORT - Constitution Update
═══════════════════════════════════════════════════════════════════════════

Version Change: NONE → 1.0.0
Change Type: MAJOR (Initial constitution creation)

Modified Principles:
  - NEW: I. Code Quality & Standards
  - NEW: II. Test-Driven Development (TDD)
  - NEW: III. User Experience Consistency
  - NEW: IV. Performance & Optimization

Added Sections:
  - Core Principles (4 principles defined)
  - Python-Specific Requirements
  - Development Workflow
  - Governance

Removed Sections: N/A (initial creation)

Template Updates:
  ✅ plan-template.md - Constitution Check section will reference new principles
  ✅ spec-template.md - Requirements section aligns with quality principles
  ✅ tasks-template.md - Task organization reflects TDD and phased workflow
  ⚠️ agent-file-template.md - Review for alignment with quality standards
  ⚠️ checklist-template.md - Review for alignment with quality standards

Follow-up TODOs: None

═══════════════════════════════════════════════════════════════════════════
-->

## Core Principles

### I. Code Quality & Standards

All code produced MUST meet the following non-negotiable quality standards:

- **Maintainability**: Code must be clear, well-documented, and follow consistent patterns
- **Modularity**: Functionality must be broken into focused, reusable components
- **Documentation**: Every public interface must have clear docstrings explaining purpose, parameters, and return values
- **Style Compliance**: Code must pass all configured linting and formatting checks before commit
- **Type Safety**: When applicable, use type hints/annotations to improve code clarity and catch errors early

**Rationale**: Quality standards prevent technical debt accumulation and ensure the codebase remains accessible to all team members, reducing onboarding time and maintenance costs.

### II. Test-Driven Development (TDD)

Testing is **NON-NEGOTIABLE** and follows strict TDD principles:

- **Red-Green-Refactor**: Write tests first → Watch them fail → Implement feature → Pass tests → Refactor
- **Test Coverage**: All new code paths must have corresponding tests
- **Test Types Required**:
  - **Unit Tests**: Test individual functions/methods in isolation
  - **Integration Tests**: Test component interactions and contracts
  - **Contract Tests**: Verify API/interface boundaries match specifications
- **Pre-Commit Gate**: All tests must pass before code can be committed
- **Independent Stories**: Each user story must be independently testable as a deliverable MVP increment

**Rationale**: TDD catches bugs early, ensures requirements are met, provides living documentation, and enables confident refactoring. Tests written after implementation often miss edge cases and don't validate actual requirements.

### III. User Experience Consistency

User-facing features must deliver consistent, predictable experiences:

- **Interface Stability**: Public APIs, CLI interfaces, and UX patterns must remain consistent within major versions
- **Error Messages**: Provide clear, actionable error messages that guide users toward resolution
- **Documentation Alignment**: User-facing docs, help text, and examples must match actual behavior
- **Accessibility**: Interfaces must be usable by diverse users (consider screen readers, keyboard navigation, etc.)
- **Feedback Mechanisms**: Provide clear progress indicators for long-running operations

**Rationale**: Consistency reduces cognitive load, accelerates user adoption, and minimizes support burden. Users trust systems that behave predictably.

### IV. Performance & Optimization

Performance characteristics must be defined, measured, and maintained:

- **Requirements Definition**: Each feature must specify performance targets (response times, throughput, resource usage)
- **Measurement**: Performance-critical paths must include instrumentation and monitoring
- **Regression Prevention**: Performance tests must be part of CI pipeline for critical paths
- **Resource Efficiency**: Code must avoid unnecessary allocations, loops, and I/O operations
- **Scalability Awareness**: Design decisions must consider how features scale with increased load/data

**Rationale**: Performance is a feature. Defining requirements upfront prevents costly rewrites and ensures user satisfaction at scale.

## Python-Specific Requirements

For Python projects, the following tools and practices are mandatory:

### Dependency & Environment Management

- **Tool**: `uv` for all dependency management and virtual environment operations
- **Rationale**: `uv` provides faster, more reliable dependency resolution than pip

### Code Quality Tooling

- **Formatter**: `ruff format` - Format all Python code before commit
- **Linter**: `ruff check` - Lint all Python code before commit
- **Configuration**: Project must include `pyproject.toml` or `ruff.toml` with explicit rule configuration

**Pre-Commit Workflow**:

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest
```

## Development Workflow

### Phase Completion Gates

Before completing any development phase (research, design, implementation):

1. **Alignment Check**: Verify all code, specs, documentation, and tests remain consistent
2. **Cross-Reference Validation**: Ensure changes are reflected across all affected artifacts
3. **Constitution Compliance**: Confirm all principles are satisfied

### Version Control Discipline

- **Commit Frequency**: Commit to version control before starting each new phase or major work block
- **Pre-Commit Validation**: If ANY code has changed, run the following before commit:
  1. Code formatting (e.g., `ruff format`)
  2. Code linting (e.g., `ruff check`)
  3. All unit tests
- **Commit Messages**: Use conventional commit format (e.g., `feat:`, `fix:`, `docs:`, `test:`)

### Ad-Hoc Changes & Consistency

When making ad-hoc changes outside the standard workflow (e.g., CLI parameter adjustments):

1. **Pause Before Proceeding**: Offer to analyze the change for consistency impact
2. **Cross-Artifact Check**: Verify specs, docs, tests, and code all reflect the change
3. **Document Decision**: Update relevant documentation to match the new behavior

**Rationale**: Ad-hoc changes are high-risk for introducing inconsistencies. Systematic verification prevents specification drift and documentation rot.

## Governance

### Authority

This constitution supersedes all other development practices and guidelines. All pull requests, code reviews, and planning documents must verify compliance with these principles.

### Amendment Process

1. **Proposal**: Document the proposed change and rationale
2. **Impact Analysis**: Identify affected templates, specs, and code
3. **Migration Plan**: Define steps to bring existing work into compliance
4. **Approval**: Secure stakeholder approval before implementation
5. **Version Update**: Increment constitution version according to semantic versioning

### Versioning Policy

- **MAJOR** (X.0.0): Backward-incompatible principle removals or redefinitions
- **MINOR** (x.Y.0): New principles added or existing principles materially expanded
- **PATCH** (x.y.Z): Clarifications, wording improvements, typo fixes

### Compliance Review

- Constitution compliance must be verified at spec, plan, and implementation phases
- Violations must be justified in writing if they cannot be resolved
- Patterns of non-compliance trigger constitution review for practical feasibility

### Runtime Guidance

For day-to-day development guidance incorporating these principles, refer to `.specify/templates/agent-file-template.md` (auto-generated from feature plans and active technologies).

**Version**: 1.0.0 | **Ratified**: 2026-01-31 | **Last Amended**: 2026-01-31
