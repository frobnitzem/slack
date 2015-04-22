/*
 *    map version 1.0.0
 *
 *    ANSI C hash table for fix-sized byte strings
 *
 *        Version history:
 *        1.0.0 - initial release
 *        2.0.0 - changed function prefix from strmap to sm to ensure
 *            ANSI C compatibility 
 *        2.0.1 - improved documentation 
 *
 *        strmap.c -> map.c 1.0.0
 *          - replaced name enum -> iter
 *          - replaced char strings -> fixed-size byte strings
 *          - changed behavior of map_put to store location,
 *            not copy and to return previous value if there is one.
 *          - changed behavior of map_get to return
 *            location or NULL, rather than copy.
 *          - removed const qualifier on map_iter (strmap_enum)
 *          - removed map_exists (duplicates map_get)
 *          - renamed map_new -> map_ctor
 *          - renamed map_destroy -> map_dtor and changed type to Map **
 *
 *    map.c
 *
 *    Copyright (c) 2009, 2011, 2013 Per Ola Kristensson.
 *    Copyright (c) 2013 David M. Rogers
 *
 *    Per Ola Kristensson <pok21@cam.ac.uk> 
 *    Inference Group, Department of Physics
 *    University of Cambridge
 *    Cavendish Laboratory
 *    JJ Thomson Avenue
 *    CB3 0HE Cambridge
 *    United Kingdom
 *
 *    David M. Rogers <predictivestatmech@gmail.com>
 *    Nonequilibrium Stat. Mech. Research Group
 *    Department of Chemistry
 *    University of South Florida
 *
 *    This file is part of cmap.
 *
 *    cmap is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    cmap is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with cmap.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <lib/map.h>

typedef struct Pair Pair;

typedef struct Bucket Bucket;

struct Pair {
    void *key; // memcpy-ed from callers (I manage)
    void *value; // copied from callers (they manage)
};

struct Bucket {
    unsigned int count;
    Pair *pairs;
};

struct Map {
    unsigned int count;
    size_t width;
    Bucket *buckets;
};

static Pair *get_pair(Bucket *bucket, const void *key, size_t width);
static unsigned long hash(const void *str, size_t width);

Map *map_ctor(unsigned int capacity, size_t width)
{
    Map *map;
    
    map = malloc(sizeof(Map));
    if (map == NULL) {
        return NULL;
    }
    map->width = width;
    map->count = capacity;
    map->buckets = calloc(map->count, sizeof(Bucket));
    if (map->buckets == NULL) {
        free(map);
        return NULL;
    }
    return map;
}

void *map_get(const Map *map, const void *key) {
    unsigned int index;
    Bucket *bucket;
    Pair *pair;

    if (map == NULL) {
        return NULL;
    }
    if (key == NULL) {
        return NULL;
    }
    index = hash(key, map->width) % map->count;
    bucket = map->buckets + index;
    if( (pair = get_pair(bucket, key, map->width)) == NULL) {
        return NULL;
    }
    return pair->value;
}

void *map_put(Map *map, const void *key, void *value)
{
    unsigned int index;
    Bucket *bucket;
    Pair *tmp_pairs, *pair;
    void *tmp_value;
    void *new_key;

    if(map == NULL || key == NULL) {
        return value;
    }
    /* Get a pointer to the bucket the key string hashes to */
    index = hash(key, map->width) % map->count;
    bucket = map->buckets + index;
    /* Check if we can handle insertion by simply replacing
     * an existing value in a key-value pair in the bucket.
     */
    if( (pair = get_pair(bucket, key, map->width)) != NULL) {
        /* The bucket contains a pair that matches the provided key,
         * change the value for that pair to the new value.
         */
        tmp_value = pair->value;
        pair->value = value;
        return tmp_value;
    }
    /* Allocate space for a new key and value */
    new_key = malloc(map->width);
    if(new_key == NULL) {
        return NULL;
    }
    /* Lazily allocate space for key-value pairs.  */
    tmp_pairs = realloc(bucket->pairs, (bucket->count + 1) * sizeof(Pair));
    if (tmp_pairs == NULL) {
        free(new_key);
        return value;
    }
    bucket->pairs = tmp_pairs;
    bucket->count++;

    /* Get the last pair in the chain for the bucket */
    pair = bucket->pairs + bucket->count - 1;
    pair->key = new_key;
    pair->value = value;
    /* Copy the key and its value into the key-value pair */
    memcpy(pair->key, key, map->width);
    return NULL;
}

int map_get_count(const Map *map) {
    unsigned int i;
    unsigned int count = 0;
    Bucket *bucket;

    if (map == NULL) {
        return 0;
    }
    bucket = map->buckets;
    for(i=0; i < map->count; i++, bucket++) {
        count += bucket->count;
    }
    return count;
}

int map_iter(const Map *map, map_iter_func iter_func, void *obj) {
    unsigned int i, j, n, m;
    unsigned int count;
    Bucket *bucket;
    Pair *pair;

    if (map == NULL || iter_func == NULL) {
        return -1;
    }
    bucket = map->buckets;
    n = map->count;
    i = 0;
    while (i < n) {
        pair = bucket->pairs;
        m = bucket->count;
        j = 0;
        while (j < m) {
            iter_func(pair->key, pair->value, obj);
            count++;
            pair++;
            j++;
        }
        bucket++;
        i++;
    }
    return count;
}
void map_dtor(Map **map) {
    unsigned int i, j, n, m;
    Bucket *bucket;
    Pair *pair;

    if (*map == NULL) {
        return;
    }
    bucket = (*map)->buckets;
    n = (*map)->count;
    i = 0;
    while (i < n) {
        pair = bucket->pairs;
        m = bucket->count;
        j = 0;
        while(j < m) {
            free(pair->key);
            pair++;
            j++;
        }
        free(bucket->pairs);
        bucket++;
        i++;
    }
    free((*map)->buckets);
    free(*map);
    *map = NULL;
}


/*
 * Returns a pair from the bucket that matches the provided key,
 * or null if no such pair exist.
 */
static Pair *get_pair(Bucket *bucket, const void *key, size_t width) {
    unsigned int i;
    Pair *pair;

    for(i=0,pair=bucket->pairs; i <  bucket->count; i++,pair++) {
        if(pair->key != NULL) {
            if (memcmp(pair->key, key, width) == 0) {
                return pair;
            }
        }
    }
    return NULL;
}

/*
 * Returns a hash code for the provided string.
 */
static unsigned long hash(const void *str, size_t width)
{
    unsigned long hash = 5381;
    const char *c = str;
    size_t i;

    for(i=0; i<width; i++,c++) {
        hash = ((hash << 5) + hash) + *c;
    }
    return hash;
}
