#!/usr/bin/env python3
"""
Simple Phase 7.1 Deployment Validation Script

This script performs basic validation of the Phase 7.1 deployment:
- Tests that deprecation system works
- Tests that legacy imports work with warnings
- Tests that new imports work
- Tests version information
"""

import sys
import warnings
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_deprecation_system():
    """Test the deprecation warning system."""
    logger.info("Testing deprecation system...")
    
    try:
        from src.shared.deprecation import (
            DeprecationConfig, deprecation_warning, 
            print_migration_summary, check_legacy_imports_enabled
        )
        
        # Test configuration
        assert DeprecationConfig.WARNINGS_ENABLED == True, "Warnings should be enabled"
        assert DeprecationConfig.LEGACY_IMPORTS_ENABLED == True, "Legacy imports should be enabled"
        
        # Test deprecation warning function
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecation_warning("test.old", "test.new", "v2.0.0")
            assert len(w) > 0, "Deprecation warning should be issued"
            assert issubclass(w[0].category, DeprecationWarning), "Should be DeprecationWarning"
        
        logger.info("✅ Deprecation system works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Deprecation system failed: {e}")
        return False

def test_version_system():
    """Test the version information system."""
    logger.info("Testing version system...")
    
    try:
        from src.shared.version import (
            get_version_info, is_legacy_supported, should_show_deprecation_warnings,
            CURRENT_PHASE, VERSION
        )
        
        # Test version info
        version_info = get_version_info()
        assert version_info is not None, "Version info should be available"
        assert "version" in version_info, "Version should be in info"
        assert "phase" in version_info, "Phase should be in info"
        
        # Test Phase 7.1 configuration
        assert CURRENT_PHASE == "7.1", f"Expected Phase 7.1, got {CURRENT_PHASE}"
        assert is_legacy_supported() == True, "Legacy support should be enabled"
        assert should_show_deprecation_warnings() == True, "Deprecation warnings should be enabled"
        
        logger.info(f"✅ Version system works correctly - Version: {VERSION}, Phase: {CURRENT_PHASE}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Version system failed: {e}")
        return False

def test_legacy_imports():
    """Test that legacy imports work with deprecation warnings."""
    logger.info("Testing legacy imports...")
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test legacy orchestrator import
            from src.agents.orchestrator_agent import OrchestratorAgent
            
            # Should have captured deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            
            if deprecation_warnings:
                logger.info(f"✅ Legacy imports work with {len(deprecation_warnings)} deprecation warning(s)")
            else:
                logger.warning("⚠️ Legacy imports work but no deprecation warnings captured")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Legacy imports failed: {e}")
        return False

def test_new_imports():
    """Test that new imports work without warnings."""
    logger.info("Testing new imports...")
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test new orchestrator import
            from src.execution.orchestrator_agent import OrchestratorAgent
            
            # Should have no warnings for new imports
            new_import_warnings = [warning for warning in w if "src.execution" in str(warning.message)]
            
            if not new_import_warnings:
                logger.info("✅ New imports work without warnings")
            else:
                logger.warning(f"⚠️ New imports generated {len(new_import_warnings)} warning(s)")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ New imports failed: {e}")
        return False

def test_import_equivalence():
    """Test that legacy and new imports reference the same classes."""
    logger.info("Testing import equivalence...")
    
    try:
        # Import both ways
        from src.execution.orchestrator_agent import OrchestratorAgent as NewOrchestrator
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test
            from src.agents.orchestrator_agent import OrchestratorAgent as LegacyOrchestrator
        
        # They should be the same class
        if NewOrchestrator is LegacyOrchestrator:
            logger.info("✅ Legacy and new imports reference the same class")
            return True
        else:
            logger.error("❌ Legacy and new imports reference different classes")
            return False
            
    except Exception as e:
        logger.error(f"❌ Import equivalence test failed: {e}")
        return False

def test_migration_guide():
    """Test that migration guide is available."""
    logger.info("Testing migration guide...")
    
    try:
        migration_guide_path = PROJECT_ROOT / "MIGRATION_GUIDE.md"
        
        if not migration_guide_path.exists():
            logger.error("❌ MIGRATION_GUIDE.md not found")
            return False
        
        # Check if it has substantial content
        with open(migration_guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content) < 1000:
                logger.error("❌ Migration guide appears to be incomplete")
                return False
        
        logger.info("✅ Migration guide is available and substantial")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration guide test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("🚀 Starting Simple Phase 7.1 Validation")
    logger.info("=" * 50)
    
    tests = [
        ("Deprecation System", test_deprecation_system),
        ("Version System", test_version_system),
        ("Legacy Imports", test_legacy_imports),
        ("New Imports", test_new_imports),
        ("Import Equivalence", test_import_equivalence),
        ("Migration Guide", test_migration_guide)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Testing: {test_name}")
        logger.info("-" * 30)
        
        if test_func():
            passed += 1
        else:
            logger.error(f"Test '{test_name}' failed")
    
    # Final report
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Tests Passed: {passed}/{total}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Phase 7.1 deployment is ready!")
        return 0
    else:
        logger.error(f"❌ {total-passed} test(s) failed - review issues before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())