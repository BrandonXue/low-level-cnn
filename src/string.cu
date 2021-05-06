#include <ctype.h>
#include <string.h>
#include <stdio.h>

#include <string.cu.h>

/* Tokens
struct Tokens {
    char **data;
    int count;
    int size;
};
*/

/**
 * Tokens Constructor
 */
Tokens* Tokens_create() {
    Tokens *toks = (Tokens*)malloc(sizeof(Tokens));
    toks->count = 0;
    toks->size = 4;
    toks->data = (char**)malloc(toks->size * sizeof(char*));
    return toks;
}

/**
 * Tokens Destructor
 */
Tokens* Tokens_destroy(Tokens *toks) {
    for (int i = 0; i < toks->count; ++i) {
        free(toks->data[i]);
        toks->data[i] = NULL;
    }
    free(toks->data);
    toks->data = NULL;
    free(toks);
    toks = NULL;
    return toks;
}

/**
 * Appends a copy of new_tok to toks.
 */
bool Tokens_append(Tokens *toks, const char *new_tok) {
    if (toks == NULL || toks->data == NULL) {
        return false;
    }
    // Reallocate memory if needed
    if (toks->count == toks->size) {
        toks->size *= 2;
        toks->data = (char**)realloc(toks->data, toks->size * sizeof(char*));
    }
    int len = strlen(new_tok) + 1;
    toks->data[toks->count] = (char*)malloc(len * sizeof(char));
    strncpy(toks->data[toks->count], new_tok, len);

    // Change newline to null terinator
    if (toks->data[toks->count][len-2] == '\n') {
        toks->data[toks->count][len-2] = '\0';
    }

    toks->count += 1;
    return true;
}

/**
 * Getter for convenience
 */
int Tokens_get_count(Tokens *toks) {
    if (toks == NULL) {
        return 0;
    }
    return toks->count;
}

/**
 * Getter for convenience
 */
char* Tokens_at(Tokens *toks, int index) {
    if (toks == NULL || index >= toks->count) {
        return NULL;
    }
    return toks->data[index];
}

/**
 * Check if a token matches any of the tokens in toks.
 */
bool Tokens_match_at(Tokens *toks, int index, const char *str) {
    // Tidy up leading whitespace in str
    while (isspace(*str) && strlen(str) > 0) {
        str = str + 1;
    }
    // Terminate str at the first whitespace
    int newlen = strlen(str);
    int i;
    for (i = 0; i < newlen; ++i) {
        if (isspace(str[i])) {
            break;
        }
    }
    char buf[newlen+1];
    strncpy(buf, str, i);
    buf[i] = '\0';
    // Make sure all pointers and indices are valid
    if (toks == NULL || toks->data == NULL || index >= toks->count) {
        return false;
    }
    // Compare the strings
    //printf("Comparing %s with %s\n", toks->data[index], buf);
    return strcmp(toks->data[index], buf) == 0;
}

/**
 * Populate Tokens with a string.
 */
bool Tokens_fetch(Tokens *toks, const char *line) {
    if (toks == NULL || toks->data == NULL || line == NULL) {
        return false;
    }
    int len = strlen(line);
    char buf[128];
    int buf_i = 0;
    for (int i = 0; i < len; ++i) {
        // On detect whitespace
        if (isspace(line[i])) {
            // If buffer not empty, create token
            if (buf_i > 0) {
                buf[buf_i] = '\0';
                Tokens_append(toks, buf);
                buf_i = 0;
            }
            // Continue eating other whitespace in between tokens
            while (i+1 < len && isspace(line[i+1])) {
                ++i;
            }
        }
        // On detect other characters, add them to buffer
        else {
            buf[buf_i] = line[i];
            ++buf_i;
        }
    }
    // On null terminate, buffer may contain last token. If so, create a token
    if (buf_i != 0) {
        buf[buf_i] = '\0';
        Tokens_append(toks, buf);
    }
    return true;
}
