# Collaboration Workflow Guide

This guide helps you safely merge files from your collaborator without losing your work.

## Method 1: Full Directory Merge (Recommended)

When your collaborator sends you a folder with multiple files:

1. **Download their files** to a temporary location (e.g., `C:\Downloads\collaborator_files\`)

2. **Run the merge script**:
   ```powershell
   .\scripts\merge_collaborator_files.ps1 -DownloadPath "C:\Downloads\collaborator_files\"
   ```

3. **Follow the interactive prompts** for each file:
   - `d` - View differences
   - `a` - Accept their version
   - `k` - Keep your version  
   - `m` - Manual merge in VS Code
   - `s` - Skip for now

4. **Test your changes** after merging

5. **Commit and push**:
   ```powershell
   git add .
   git commit -m "Merged changes from collaborator"
   git push
   ```

## Method 2: Single File Comparison

For comparing individual files:

```powershell
.\scripts\quick_compare.ps1 "C:\Downloads\their_file.py" "src\agents\data_agent.py"
```

## Method 3: Manual VS Code Diff

1. **Don't overwrite** your file directly
2. **Save their file** with a different name (e.g., `data_agent_collaborator.py`)
3. **Open VS Code diff**:
   ```powershell
   code --diff "src\agents\data_agent.py" "src\agents\data_agent_collaborator.py"
   ```
4. **Review changes** side by side
5. **Manually merge** what you want to keep
6. **Delete the temporary file** when done

## Method 4: Git Branch Approach (Advanced)

If your collaborator can use Git:

1. **Create a branch** for their changes:
   ```powershell
   git checkout -b collaborator-changes
   ```

2. **Add their files** to this branch
   ```powershell
   git add .
   git commit -m "Collaborator changes"
   ```

3. **Switch back to main** and merge:
   ```powershell
   git checkout main
   git merge collaborator-changes
   ```

4. **Resolve conflicts** if any, then push

## Best Practices

### Before Merging:
- ✅ **Always commit your current work** first
- ✅ **Create a backup** of important files
- ✅ **Test the merged code** before pushing

### During Review:
- 🔍 **Check for breaking changes**
- 🔍 **Verify new dependencies** in requirements.txt
- 🔍 **Look for configuration changes**
- 🔍 **Review any new files** carefully

### After Merging:
- ✅ **Run tests** if you have them
- ✅ **Check that the application still works**
- ✅ **Update documentation** if needed
- ✅ **Commit with descriptive message**

## Emergency Recovery

If something goes wrong:

```powershell
# See recent commits
git log --oneline -10

# Revert to previous commit (replace COMMIT_HASH)
git reset --hard COMMIT_HASH

# Or just undo last commit (keeps files)
git reset --soft HEAD~1
```

## File Naming Convention for Collaborator

Ask your collaborator to use this naming pattern:
- `filename_YYYYMMDD.py` (e.g., `data_agent_20241201.py`)
- Or create a folder: `collaborator_changes_YYYYMMDD/`

This makes it clear which files are new versions.

## Tools You'll Need

- **VS Code** (for diff viewing and manual merging)
- **PowerShell** (for running the scripts)
- **Git** (for version control and recovery)

## Troubleshooting

**Q: Script won't run?**
A: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first

**Q: VS Code diff not working?**
A: Make sure VS Code is installed and in your PATH, or use full path to code.exe

**Q: Lost changes?**
A: Use `git reflog` to find your previous commits and recover them