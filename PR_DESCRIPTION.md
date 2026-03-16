Title: feat: Add CIFAR-100 dataset loading and dataloader creation utility

## Summary
<!--- Add some bells and whistles for PR template. --->

Add CIFAR-100 dataset loading and dataloader creation utility to the project. This PR sets up the data processing pipeline for image classification tasks using the CIFAR-100 dataset.

### Why
<!--- Clearly define the issue or problem that your changes address.
Describe what is currently not working as expected or what feature is missing. --->

The project requires a standardized and reusable way to load, preprocess, and split the CIFAR-100 dataset for training, validation, and testing. Currently, there was no centralized utility for this, making it difficult to maintain consistency across different training scripts.

### What
<!--- Provide a high-level overview of what has been modified, added, or removed in the codebase.
This could include new features, bug fixes, refactoring efforts, or performance optimizations. --->

- **New Utility**: Created `src/data/image_dataset.py` with the `get_image_dataloaders` function.
- **Dataset Support**: Integrated torchvision's `CIFAR100` dataset.
- **Preprocessing**: Implemented a transformation pipeline including resizing, tensor conversion, and normalization using CIFAR-100 specific statistics.
- **Data Splitting**: Added logic to split the training set into training (90%) and validation (10%) sets.
- **Project Structure**: Added `src/data/__init__.py` to enable package imports.
- **Maintenance**: Updated `.gitignore` to exclude output directories, Python cache files, IDE settings, and logs.

### Solution
<!--- Describe the architectural or design decisions you made while implementing the changes.
Explain the thought process behind your approach and how it aligns with best practices or existing patterns in the codebase. --->

The implementation uses a centralized `get_image_dataloaders` function that returns a dictionary containing the `DataLoader` objects and metadata (class names, number of classes). This approach follows standard PyTorch practices and ensures that the same preprocessing logic is applied consistently. The use of `random_split` ensures a reproducible and fair validation set from the training data.

## Types of Changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply --->
- [ ] ❌ Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x]  New feature (non-breaking change which adds functionality)
- [ ]  Bug fix (non-breaking change which fixes an issue)
- [ ]  Performance optimization (non-breaking change which addresses a performance issue)
- [ ]  Refactor (non-breaking change which does not change existing behavior or add new functionality)
- [ ]  Library update (non-breaking change that will update one or more libraries to newer versions)
- [ ]  Documentation (non-breaking change that doesn't change code behavior, can skip testing)
- [ ] ✅ Test (non-breaking change related to testing)
- [ ]  Security awareness (changes that effect permission scope, security scenarios)

## Test Plan
<!--- Please input steps on how to test this PR, including evidence in the form of captured images or videos. If this is not necessary, provide the reason why. --->

1. **Verification of Dataloaders**:
   - Instantiate the dataloaders using `get_image_dataloaders()`.
   - Verify that `train_loader`, `val_loader`, and `test_loader` are correctly created.
   - Fetch a batch from each loader to ensure dimensions match the expected `(batch_size, 3, image_size, image_size)`.
   - Confirm that the number of classes and class names match the CIFAR-100 specification (100 classes).

2. **Integration Test**:
   - Run a sample script that imports `get_image_dataloaders` and prints the size of the datasets.

## Related Issues
<!--- Add a reference section for management tickets, and relevant conversations. --->
- Related to image classification setup for Assignment 1.
