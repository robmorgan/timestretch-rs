-- export_dialog.applescript
-- Automates the Ableton Live Export Audio/Video dialog via System Events.
--
-- Usage:
--   osascript export_dialog.applescript /path/to/output/folder filename_without_ext
--
-- Prerequisites:
--   - Ableton Live 11 Suite must be running with a project loaded
--   - Terminal/iTerm must have Accessibility permissions
--   - System Settings > Privacy & Security > Accessibility
--
-- Note: This is inherently fragile GUI automation. Increase delays if
-- your machine is slower or Ableton is under heavy load.

on run argv
	if (count of argv) < 2 then
		error "Usage: osascript export_dialog.applescript <output_folder> <output_filename>"
	end if

	set outputFolder to item 1 of argv
	set outputFilename to item 2 of argv

	-- Configurable delays (seconds) -- increase for slower machines
	set dialogOpenDelay to 3
	set goToFolderDelay to 1.5
	set navigationDelay to 1.5
	set typeDelay to 0.5

	tell application "Ableton Live 11 Suite"
		activate
	end tell

	delay 1

	tell application "System Events"
		tell process "Ableton Live 11 Suite"
			-- Ensure Ableton is frontmost
			set frontmost to true
			delay 0.5

			-- Open Export Audio/Video dialog: Cmd+Shift+R
			keystroke "r" using {command down, shift down}
			delay dialogOpenDelay

			-- The export dialog should now be open.
			-- Click the "Export" button to trigger the save dialog.
			-- First, we need to make sure the format settings are correct.

			-- Click Export (or press Enter if Export is the default button)
			keystroke return
			delay dialogOpenDelay

			-- Now we should be in the save file dialog.
			-- Navigate to the output folder using Cmd+Shift+G (Go to Folder)
			keystroke "g" using {command down, shift down}
			delay goToFolderDelay

			-- Type the output folder path
			keystroke outputFolder
			delay typeDelay
			keystroke return
			delay navigationDelay

			-- Select all text in the filename field and replace it
			keystroke "a" using {command down}
			delay typeDelay
			keystroke outputFilename
			delay typeDelay

			-- Press Enter to save
			keystroke return
			delay 1

			-- If an overwrite confirmation appears, confirm it
			try
				delay 1
				keystroke return
			end try
		end tell
	end tell
end run
