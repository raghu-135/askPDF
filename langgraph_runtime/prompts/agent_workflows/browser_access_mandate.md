# Browser Workbench Mandate

Interactive browser tools are ENABLED for this session (navigate, read, save captured pages to the thread).

## When to use the `web` route and browser tools

- Choose route `web` when the answer requires loading or reading a specific live web page, not merely a keyword web search.
- Set `tool_name` to `browser_navigate` and `query` to the target URL (or a concise navigation intent that includes the URL).
- Prefer `browser_read` after navigation to extract page text; use `browser_save_to_thread` when the user needs the page kept in the thread.
- Do not use `search_web` when the user pointed at a URL or named a page that must be opened in the workbench browser.

## Approval

In ask mode, navigation may pause for user approval once per run; continue after approval with the same URL.