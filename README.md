# Entity Masking

Illustration of entity masking functions for use with sensitive online forum data. To disseminate results without divulging things like:

 - names
 - email addresses
 - home addresses
 - social security numbers
 - bank account numbers

It also does fuzzy name matching, so if 'Andy Wheeler' is mentioned one place, and 'Andrew Wheeler' is mentioned another, it will match those two inputs to the same masked token.

Andy Wheeler
