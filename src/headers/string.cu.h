

struct Tokens {
    char **data;
    int count;
    int size;
};

Tokens* Tokens_create();

Tokens* Tokens_destroy(Tokens*);

bool Tokens_append(Tokens*, const char*);

int Tokens_get_count(Tokens*);

char* Tokens_at(Tokens*, int);

bool Tokens_match_at(Tokens*, int, const char*);

bool Tokens_fetch(Tokens*, const char*);
