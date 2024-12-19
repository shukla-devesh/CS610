#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>

#ifdef USE_TBB
#include <tbb/concurrent_hash_map.h>
#else
#include <pthread.h>
#endif

typedef struct {
  uint32_t key;
  uint32_t value;
} KeyValue;

#ifdef USE_TBB
typedef tbb::concurrent_hash_map<uint32_t, uint32_t> HashTable;

void batch_insert(HashTable *ht, KeyValue* keyValues, uint32_t count, bool *result) {
    
    for (size_t i = 0; i < count; i++) {
        auto res = ht->insert(std::make_pair(keyValues[i].key, keyValues[i].value));
        result[i] = res;
    }
}

void batch_delete(HashTable *ht, uint32_t* keys, uint32_t count, bool *result) {
    for (size_t i = 0; i < count; i++) {
        HashTable::accessor acc;
        result[i] = ht->find(acc, keys[i]);
        if (result[i]) {
            ht->erase(acc);
        }
    }
}

void batch_lookup(HashTable *ht, uint32_t* keys, uint32_t count, uint32_t *result) {
    for (size_t i = 0; i < count; i++) {
        HashTable::const_accessor acc;
        if (ht->find(acc, keys[i])) {
            result[i] = acc->second;
        } else {
            result[i] = static_cast<uint32_t>(-1);
        }
    }
}

void print_hash_table(HashTable *ht) {
    for (auto it = ht->begin(); it != ht->end(); ++it) {
        printf("Key=%u, Value=%u\n", it->first, it->second);
    }
}

#else
typedef struct {
    KeyValue *entries;
    pthread_mutex_t *locks; // One mutex per bucket for fine-grained locking
    size_t size;
    size_t count;
} HashTable;

void init_hash_table(HashTable *ht, size_t size) {
    ht->size = size;
    ht->count = 0;
    ht->entries = (KeyValue*)calloc(size, sizeof(KeyValue));
    for (size_t i = 0; i < size; i++) {
        ht->entries[i].key = -1;
        ht->entries[i].value = -1;
    }
    ht->locks = (pthread_mutex_t*)malloc(size * sizeof(pthread_mutex_t));
    
    for (size_t i = 0; i < size; i++) {
        pthread_mutex_init(&ht->locks[i], NULL);
    }
}

void free_hash_table(HashTable *ht) {
    for (size_t i = 0; i < ht->size; i++) {
        pthread_mutex_destroy(&ht->locks[i]);
    }
    free(ht->locks);
    free(ht->entries);
}

// Modular hashing
// uint32_t primary_hash(uint32_t key, size_t size) {
//     return key % size;
// }

// Bitwise hashing
uint32_t primary_hash(uint32_t key, size_t size) {
    int shift = 4;
    int raw_hash = ((key ^ (key >> shift)) & size);
    return raw_hash % size;
}


// Simple modulus
// uint32_t secondary_hash(uint32_t key, size_t size) {
//     return 1 + (key % (size - 1));
// }

// Odd Hash function
uint32_t secondary_hash(uint32_t key, size_t size) {
    return (key / size) % size | 1;
}

void batch_insert(HashTable *ht, KeyValue* keyValues, uint32_t count, bool *result) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        uint32_t key = keyValues[i].key;
        uint32_t value = keyValues[i].value;
        size_t idx = primary_hash(key, ht->size);
        size_t step = secondary_hash(key, ht->size);

        while (1) {
            pthread_mutex_lock(&ht->locks[idx]);
            if (!ht->entries[idx].key != -1 || ht->entries[idx].key == key) {
                // if (!ht->entries[idx].is_occupied) ht->count++;
                ht->entries[idx].key = key;
                ht->entries[idx].value = value;
                // ht->entries[idx].is_occupied = true;
                result[i] = true;
                pthread_mutex_unlock(&ht->locks[idx]);
                break;
            }
            pthread_mutex_unlock(&ht->locks[idx]);
            idx = (idx + step) % ht->size;
        }
    }
}


void batch_delete(HashTable *ht, uint32_t* keys, uint32_t count, bool *result) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        uint32_t key = keys[i];
        size_t idx = primary_hash(key, ht->size);
        size_t step = secondary_hash(key, ht->size);

        result[i] = false;  // Assume not found initially

        while (1) {
            pthread_mutex_lock(&ht->locks[idx]);
            if (ht->entries[idx].key != -1 && ht->entries[idx].key == key) {
                // ht->entries[idx].is_occupied = false;
                result[i] = true;  // Deletion successful
                ht->count--;
                pthread_mutex_unlock(&ht->locks[idx]);
                break;
            }
            if (!ht->entries[idx].key != -1) {  // Key not found
                pthread_mutex_unlock(&ht->locks[idx]);
                break;
            }
            pthread_mutex_unlock(&ht->locks[idx]);
            idx = (idx + step) % ht->size;
        }
    }
}

void batch_lookup(HashTable *ht, uint32_t* keys, uint32_t count, uint32_t *result) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        uint32_t key = keys[i];
        size_t idx = primary_hash(key, ht->size);
        size_t step = secondary_hash(key, ht->size);

        result[i] = -1;  // Default value for not found

        while (1) {
            pthread_mutex_lock(&ht->locks[idx]);
            if (ht->entries[idx].key != -1 && ht->entries[idx].key == key) {
                result[i] = ht->entries[idx].value;  // Key found
                pthread_mutex_unlock(&ht->locks[idx]);
                break;
            }
            if (!ht->entries[idx].key != -1) {  // Key not found
                pthread_mutex_unlock(&ht->locks[idx]);
                break;
            }
            pthread_mutex_unlock(&ht->locks[idx]);
            idx = (idx + step) % ht->size;
        }
    }
}

void print_hash_table(HashTable *ht) {
    for (size_t i = 0; i < ht->size; i++) {
        if (ht->entries[i].key != -1) {
            printf("Index %zu: Key=%u, Value=%u\n", i, ht->entries[i].key, ht->entries[i].value);
        }
    }
}

#endif